package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func minimalGLB() []byte {
	jsonChunk := []byte(`{"asset":{"version":"2.0"}}`)
	for len(jsonChunk)%4 != 0 {
		jsonChunk = append(jsonChunk, ' ')
	}
	result := make([]byte, 20+len(jsonChunk))
	binary.LittleEndian.PutUint32(result[0:4], 0x46546c67)
	binary.LittleEndian.PutUint32(result[4:8], 2)
	binary.LittleEndian.PutUint32(result[8:12], uint32(len(result)))
	binary.LittleEndian.PutUint32(result[12:16], uint32(len(jsonChunk)))
	binary.LittleEndian.PutUint32(result[16:20], 0x4e4f534a)
	copy(result[20:], jsonChunk)
	return result
}

type copyAnimator struct {
	delay time.Duration
}

func (a copyAnimator) Animate(ctx context.Context, source, output string, _ animationOptions) (string, error) {
	if a.delay > 0 {
		select {
		case <-time.After(a.delay):
		case <-ctx.Done():
			return "", ctx.Err()
		}
	}
	input, err := os.Open(source)
	if err != nil {
		return "", err
	}
	defer input.Close()
	result, err := os.OpenFile(output, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return "", err
	}
	_, copyErr := io.Copy(result, input)
	closeErr := result.Close()
	if copyErr != nil {
		return "", copyErr
	}
	return "test-vulkan", closeErr
}

func uploadRequest(t *testing.T, s *server, name string, contents []byte) *httptest.ResponseRecorder {
	t.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("model", name)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := part.Write(contents); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, "/api/models", &body)
	request.Header.Set("Content-Type", writer.FormDataContentType())
	response := httptest.NewRecorder()
	s.handleModels(response, request)
	return response
}

func queueAnimation(t *testing.T, s *server, modelID, prompt string) string {
	t.Helper()
	values := url.Values{"prompt": {prompt}, "steps": {"2"}, "max_vertices": {"64"}}
	request := httptest.NewRequest(http.MethodPost, "/api/model/"+modelID+"/animations", strings.NewReader(values.Encode()))
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	response := httptest.NewRecorder()
	s.handleModel(response, request)
	if response.Code != http.StatusAccepted {
		t.Fatalf("queue animation: status %d: %s", response.Code, response.Body.String())
	}
	if got := response.Header().Get("Content-Type"); !strings.HasPrefix(got, "application/json") {
		t.Fatalf("queue response content type = %q", got)
	}
	var result map[string]string
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	return result["job"]
}

func waitForJob(t *testing.T, s *server, id string) animationRecord {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		s.mu.RLock()
		job := s.jobs[id]
		if job != nil && (job.State == "done" || job.State == "error") {
			result := publicAnimation(job)
			s.mu.RUnlock()
			return result
		}
		s.mu.RUnlock()
		time.Sleep(time.Millisecond)
	}
	t.Fatal("job did not finish")
	return animationRecord{}
}

func TestGalleryPersistsMultipleAnimations(t *testing.T) {
	store, err := openStore(t.TempDir(), 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	s, err := newServer(store, copyAnimator{delay: 5 * time.Millisecond}, time.Second, "vulkan", "model.gguf")
	if err != nil {
		t.Fatal(err)
	}
	go s.worker()
	t.Cleanup(func() { close(s.queue) })

	upload := uploadRequest(t, s, "../friendly dragon.glb", minimalGLB())
	if upload.Code != http.StatusCreated {
		t.Fatalf("upload: status %d: %s", upload.Code, upload.Body.String())
	}
	if got := upload.Header().Get("Content-Type"); !strings.HasPrefix(got, "application/json") {
		t.Fatalf("upload response content type = %q", got)
	}
	var model modelSummary
	if err := json.Unmarshal(upload.Body.Bytes(), &model); err != nil {
		t.Fatal(err)
	}
	if model.Name != "friendly dragon.glb" || !validID(model.ID) {
		t.Fatalf("unexpected uploaded model: %+v", model)
	}

	jobs := []string{
		queueAnimation(t, s, model.ID, "Flap both wings"),
		queueAnimation(t, s, model.ID, "Turn and bow"),
	}
	for _, id := range jobs {
		job := waitForJob(t, s, id)
		if job.State != "done" || job.Backend != "test-vulkan" {
			t.Fatalf("unexpected job: %+v", job)
		}
	}

	restored, err := store.restore()
	if err != nil {
		t.Fatal(err)
	}
	if len(restored) != 1 || len(restored[0].Animations) != 2 {
		t.Fatalf("restore returned %d models and %d animations", len(restored), len(restored[0].Animations))
	}
	for _, animation := range restored[0].Animations {
		if err := validateGLBFile(animation.OutputPath, 1<<20); err != nil {
			t.Fatalf("restored output: %v", err)
		}
	}
}

func TestConcurrentJobReadsAreRaceFree(t *testing.T) {
	store, err := openStore(t.TempDir(), 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	s, err := newServer(store, copyAnimator{delay: 20 * time.Millisecond}, time.Second, "vulkan", "model.gguf")
	if err != nil {
		t.Fatal(err)
	}
	go s.worker()
	t.Cleanup(func() { close(s.queue) })
	upload := uploadRequest(t, s, "mesh.glb", minimalGLB())
	var model modelSummary
	if err := json.Unmarshal(upload.Body.Bytes(), &model); err != nil {
		t.Fatal(err)
	}
	id := queueAnimation(t, s, model.ID, "Spin safely")
	var readers sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		readers.Add(1)
		go func() {
			defer readers.Done()
			for iteration := 0; iteration < 50; iteration++ {
				request := httptest.NewRequest(http.MethodGet, "/api/job/"+id, nil)
				response := httptest.NewRecorder()
				s.handleJob(response, request)
				if response.Code != http.StatusOK {
					t.Errorf("job read status = %d", response.Code)
					return
				}
			}
		}()
	}
	readers.Wait()
	if job := waitForJob(t, s, id); job.State != "done" {
		t.Fatalf("job state = %s: %s", job.State, job.Error)
	}
}

func TestRejectsMalformedInputs(t *testing.T) {
	store, err := openStore(t.TempDir(), 64)
	if err != nil {
		t.Fatal(err)
	}
	s, err := newServer(store, copyAnimator{}, time.Second, "cpu", "model.gguf")
	if err != nil {
		t.Fatal(err)
	}
	if response := uploadRequest(t, s, "bad.glb", []byte("not a glb")); response.Code != http.StatusBadRequest {
		t.Fatalf("malformed GLB status = %d", response.Code)
	}
	tooLarge := append(minimalGLB(), make([]byte, 80)...)
	if response := uploadRequest(t, s, "large.glb", tooLarge); response.Code != http.StatusBadRequest {
		t.Fatalf("large upload status = %d", response.Code)
	}

	request := httptest.NewRequest(http.MethodPost, "/parse", strings.NewReader("prompt=move&guidance=NaN"))
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	if err := request.ParseForm(); err != nil {
		t.Fatal(err)
	}
	if _, err := parseAnimationOptions(request); err == nil {
		t.Fatal("NaN guidance was accepted")
	}
	if validThumbnail("data:image/png;base64," + strings.Repeat("A", maximumThumbnail*2)) {
		t.Fatal("oversized thumbnail was accepted")
	}

	outside := &modelAsset{ID: "0123456789abcdef", Directory: filepath.Join(store.root, "..", "outside")}
	if err := store.deleteModel(outside); err == nil {
		t.Fatal("unsafe model deletion was accepted")
	}
}

func FuzzValidateUploadedGLB(f *testing.F) {
	f.Add(minimalGLB())
	f.Add([]byte("glTF\x02\x00\x00\x00"))
	f.Add(make([]byte, 20))
	root := f.TempDir()
	path := filepath.Join(root, "fuzz.glb")
	f.Fuzz(func(t *testing.T, contents []byte) {
		if len(contents) > 1<<20 {
			t.Skip()
		}
		if err := os.WriteFile(path, contents, 0o600); err != nil {
			t.Fatal(err)
		}
		_ = validateGLBFile(path, 1<<20)
	})
}
