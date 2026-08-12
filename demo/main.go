// animate-any-mesh demo server: upload GLB models, queue animations, and keep a
// durable per-model gallery. Inference is delegated to the hardened aam-cli so
// the web process does not add another model-loading or FFI surface.
package main

import (
	"context"
	"embed"
	"encoding/base64"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io/fs"
	"log"
	"math"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

//go:embed web
var embeddedWeb embed.FS

// The project reuses LocalAI's existing raster mark from the neighboring demo
// projects. Keeping it base64-encoded lets every repository edit remain a text
// patch while the browser still receives the original PNG bytes.
//
//go:embed web/localai.png.b64
var localAIBase64 string

const (
	defaultUploadLimit = int64(256 << 20)
	maximumThumbnail   = 512 << 10
	maximumPromptBytes = 1024
)

type queuedAnimation struct {
	modelID     string
	animationID string
}

type server struct {
	mu       sync.RWMutex
	models   map[string]*modelAsset
	jobs     map[string]*animationRecord
	queue    chan queuedAnimation
	store    *assetStore
	animator animator
	timeout  time.Duration
	device   string
	model    string
}

type modelSummary struct {
	ID             string `json:"id"`
	Name           string `json:"name"`
	CreatedAt      int64  `json:"createdAt"`
	SizeBytes      int64  `json:"sizeBytes"`
	Thumbnail      string `json:"thumbnail,omitempty"`
	AnimationCount int    `json:"animationCount"`
}

type modelDetails struct {
	modelSummary
	Animations []animationRecord `json:"animations"`
}

func newServer(
	store *assetStore, runner animator, timeout time.Duration, device, modelPath string,
) (*server, error) {
	models, err := store.restore()
	if err != nil {
		return nil, err
	}
	s := &server{
		models: make(map[string]*modelAsset), jobs: make(map[string]*animationRecord),
		queue: make(chan queuedAnimation, 32), store: store, animator: runner,
		timeout: timeout, device: device, model: filepath.Base(modelPath),
	}
	for _, model := range models {
		s.models[model.ID] = model
		for _, animation := range model.Animations {
			s.jobs[animation.ID] = animation
		}
	}
	return s, nil
}

func (s *server) worker() {
	for queued := range s.queue {
		s.mu.Lock()
		model := s.models[queued.modelID]
		animation := s.jobs[queued.animationID]
		if model == nil || animation == nil || animation.State != "queued" {
			s.mu.Unlock()
			continue
		}
		animation.State = "running"
		sourcePath := model.SourcePath
		options := animation.Options
		s.mu.Unlock()

		temporary, outputPath, err := s.store.beginAnimation(model, animation)
		started := time.Now()
		backend := ""
		if err == nil {
			ctx, cancel := context.WithTimeout(context.Background(), s.timeout)
			backend, err = s.animator.Animate(ctx, sourcePath, outputPath, options)
			cancel()
		}
		finished := time.Now()
		saved := *animation
		saved.Backend = backend
		saved.FinishedAt = finished.UnixMilli()
		saved.DurationMS = finished.Sub(started).Milliseconds()
		if err == nil {
			err = s.store.commitAnimation(model, &saved, temporary)
		}
		if err != nil && temporary != "" {
			_ = os.RemoveAll(temporary)
		}

		s.mu.Lock()
		animation.Backend = saved.Backend
		animation.FinishedAt = saved.FinishedAt
		animation.DurationMS = saved.DurationMS
		if err != nil {
			animation.State = "error"
			animation.Error = boundedError(err)
		} else {
			animation.Directory = saved.Directory
			animation.OutputPath = saved.OutputPath
			animation.State = "done"
			model.Animations = append(model.Animations, animation)
			sort.Slice(model.Animations, func(i, j int) bool {
				return model.Animations[i].CreatedAt > model.Animations[j].CreatedAt
			})
		}
		s.mu.Unlock()
		if err != nil {
			log.Printf("animation %s failed: %v", animation.ID, err)
		} else {
			log.Printf("animation %s completed in %.2fs on %s",
				animation.ID, float64(animation.DurationMS)/1000, animation.Backend)
		}
	}
}

func boundedError(err error) string {
	value := err.Error()
	if len(value) > 4096 {
		value = value[:4096] + "…"
	}
	return value
}

func (s *server) handleModels(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		s.mu.RLock()
		models := make([]modelSummary, 0, len(s.models))
		for _, model := range s.models {
			models = append(models, summarizeModel(model))
		}
		s.mu.RUnlock()
		sort.Slice(models, func(i, j int) bool { return models[i].CreatedAt > models[j].CreatedAt })
		writeJSON(w, models)
	case http.MethodPost:
		s.handleUpload(w, r)
	default:
		w.Header().Set("Allow", "GET, POST")
		http.Error(w, "GET or POST only", http.StatusMethodNotAllowed)
	}
}

func summarizeModel(model *modelAsset) modelSummary {
	return modelSummary{
		ID: model.ID, Name: model.Name, CreatedAt: model.CreatedAt,
		SizeBytes: model.SizeBytes, Thumbnail: model.Thumbnail,
		AnimationCount: len(model.Animations),
	}
}

func (s *server) handleUpload(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, s.store.maxUpload+(1<<20))
	if err := r.ParseMultipartForm(1 << 20); err != nil {
		http.Error(w, "invalid upload: "+err.Error(), http.StatusBadRequest)
		return
	}
	if r.MultipartForm != nil {
		defer r.MultipartForm.RemoveAll()
	}
	file, header, err := r.FormFile("model")
	if err != nil {
		http.Error(w, "missing model field", http.StatusBadRequest)
		return
	}
	defer file.Close()
	asset, err := s.store.createModel(header.Filename, file)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	s.models[asset.ID] = asset
	s.mu.Unlock()
	w.Header().Set("Location", "/api/model/"+asset.ID)
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(http.StatusCreated)
	writeJSON(w, summarizeModel(asset))
}

func (s *server) handleModel(w http.ResponseWriter, r *http.Request) {
	remainder := strings.TrimPrefix(r.URL.Path, "/api/model/")
	parts := strings.Split(strings.Trim(remainder, "/"), "/")
	if len(parts) == 0 || !validID(parts[0]) {
		http.NotFound(w, r)
		return
	}
	id := parts[0]
	s.mu.RLock()
	model := s.models[id]
	s.mu.RUnlock()
	if model == nil {
		http.NotFound(w, r)
		return
	}
	if len(parts) == 1 {
		switch r.Method {
		case http.MethodGet:
			s.mu.RLock()
			details := modelDetails{modelSummary: summarizeModel(model)}
			details.Animations = make([]animationRecord, len(model.Animations))
			for index, animation := range model.Animations {
				details.Animations[index] = publicAnimation(animation)
			}
			s.mu.RUnlock()
			writeJSON(w, details)
		case http.MethodDelete:
			s.deleteModel(w, model)
		default:
			http.Error(w, "GET or DELETE only", http.StatusMethodNotAllowed)
		}
		return
	}
	if len(parts) != 2 {
		http.NotFound(w, r)
		return
	}
	switch parts[1] {
	case "source":
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			http.Error(w, "GET or HEAD only", http.StatusMethodNotAllowed)
			return
		}
		serveGLB(w, r, model.SourcePath, model.Name, model.CreatedAt)
	case "animations":
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}
		s.handleAnimate(w, r, model)
	case "thumbnail":
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}
		s.handleThumbnail(w, r, model)
	default:
		http.NotFound(w, r)
	}
}

func publicAnimation(animation *animationRecord) animationRecord {
	return animationRecord{
		ID: animation.ID, ModelID: animation.ModelID, CreatedAt: animation.CreatedAt,
		FinishedAt: animation.FinishedAt, DurationMS: animation.DurationMS,
		State: animation.State, Error: animation.Error, Backend: animation.Backend,
		Options: animation.Options,
	}
}

func (s *server) handleAnimate(w http.ResponseWriter, r *http.Request, model *modelAsset) {
	if err := r.ParseForm(); err != nil {
		http.Error(w, "invalid form", http.StatusBadRequest)
		return
	}
	options, err := parseAnimationOptions(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	id, err := randomID()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	animation := &animationRecord{
		ID: id, ModelID: model.ID, CreatedAt: time.Now().UnixMilli(),
		State: "queued", Options: options,
	}
	s.mu.Lock()
	if s.models[model.ID] != model {
		s.mu.Unlock()
		http.Error(w, "model no longer exists", http.StatusNotFound)
		return
	}
	s.jobs[id] = animation
	s.mu.Unlock()
	select {
	case s.queue <- queuedAnimation{modelID: model.ID, animationID: id}:
		w.Header().Set("Location", "/api/job/"+id)
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.Header().Set("Cache-Control", "no-store")
		w.WriteHeader(http.StatusAccepted)
		writeJSON(w, map[string]string{"job": id})
	default:
		s.mu.Lock()
		delete(s.jobs, id)
		s.mu.Unlock()
		http.Error(w, "animation queue is full", http.StatusServiceUnavailable)
	}
}

func parseAnimationOptions(r *http.Request) (animationOptions, error) {
	prompt := strings.TrimSpace(r.FormValue("prompt"))
	if prompt == "" || len(prompt) > maximumPromptBytes || !utf8.ValidString(prompt) {
		return animationOptions{}, errors.New("prompt must be valid UTF-8 between 1 and 1024 bytes")
	}
	seed, err := parseUintRange(r.FormValue("seed"), 666, 0, math.MaxUint64, "seed")
	if err != nil {
		return animationOptions{}, err
	}
	steps, err := parseIntRange(r.FormValue("steps"), 64, 2, 1024, "steps")
	if err != nil {
		return animationOptions{}, err
	}
	trajectories, err := parseIntRange(r.FormValue("trajectories"), 0, 0, 16384, "trajectories")
	if err != nil {
		return animationOptions{}, err
	}
	maxVertices, err := parseIntRange(r.FormValue("max_vertices"), 10000, 3, 16_384, "max_vertices")
	if err != nil {
		return animationOptions{}, err
	}
	guidance, err := parseFloatRange(r.FormValue("guidance"), 3, -100, 100, "guidance")
	if err != nil {
		return animationOptions{}, err
	}
	fps, err := parseFloatRange(r.FormValue("fps"), 10, 1, 120, "fps")
	if err != nil {
		return animationOptions{}, err
	}
	return animationOptions{
		Prompt: prompt, Seed: seed, Steps: steps, Guidance: guidance,
		Trajectories: trajectories, FPS: fps, MaxVertices: maxVertices,
	}, nil
}

func parseIntRange(value string, fallback, minimum, maximum int, name string) (int, error) {
	if value == "" {
		return fallback, nil
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed < minimum || parsed > maximum {
		return 0, fmt.Errorf("%s must be in [%d, %d]", name, minimum, maximum)
	}
	return parsed, nil
}

func parseUintRange(value string, fallback, minimum, maximum uint64, name string) (uint64, error) {
	if value == "" {
		return fallback, nil
	}
	parsed, err := strconv.ParseUint(value, 10, 64)
	if err != nil || parsed < minimum || parsed > maximum {
		return 0, fmt.Errorf("%s is outside its allowed range", name)
	}
	return parsed, nil
}

func parseFloatRange(value string, fallback, minimum, maximum float64, name string) (float64, error) {
	if value == "" {
		return fallback, nil
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil || math.IsNaN(parsed) || math.IsInf(parsed, 0) || parsed < minimum || parsed > maximum {
		return 0, fmt.Errorf("%s must be finite and in [%g, %g]", name, minimum, maximum)
	}
	return parsed, nil
}

func (s *server) handleThumbnail(w http.ResponseWriter, r *http.Request, model *modelAsset) {
	r.Body = http.MaxBytesReader(w, r.Body, maximumThumbnail*2)
	var request struct {
		Thumbnail string `json:"thumbnail"`
	}
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&request); err != nil || !validThumbnail(request.Thumbnail) {
		http.Error(w, "thumbnail must be a small PNG, JPEG, or WebP data URL", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	err := s.store.saveThumbnail(model, request.Thumbnail)
	s.mu.Unlock()
	if err != nil {
		http.Error(w, "save thumbnail: "+err.Error(), http.StatusInternalServerError)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func validThumbnail(value string) bool {
	prefixes := []string{"data:image/png;base64,", "data:image/jpeg;base64,", "data:image/webp;base64,"}
	encoded := ""
	for _, prefix := range prefixes {
		if strings.HasPrefix(value, prefix) {
			encoded = strings.TrimPrefix(value, prefix)
			break
		}
	}
	if encoded == "" || base64.StdEncoding.DecodedLen(len(encoded)) > maximumThumbnail {
		return false
	}
	decoded, err := base64.StdEncoding.DecodeString(encoded)
	return err == nil && len(decoded) > 0 && len(decoded) <= maximumThumbnail
}

func (s *server) handleJob(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "GET only", http.StatusMethodNotAllowed)
		return
	}
	id := strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/job/"), "/")
	if !validID(id) {
		http.NotFound(w, r)
		return
	}
	s.mu.RLock()
	animation := s.jobs[id]
	if animation == nil {
		s.mu.RUnlock()
		http.NotFound(w, r)
		return
	}
	result := publicAnimation(animation)
	s.mu.RUnlock()
	writeJSON(w, result)
}

func (s *server) handleAnimation(w http.ResponseWriter, r *http.Request) {
	id := strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/animation/"), "/")
	if !validID(id) {
		http.NotFound(w, r)
		return
	}
	s.mu.RLock()
	animation := s.jobs[id]
	if animation == nil || animation.State != "done" {
		s.mu.RUnlock()
		http.NotFound(w, r)
		return
	}
	path, createdAt, prompt := animation.OutputPath, animation.CreatedAt, animation.Options.Prompt
	s.mu.RUnlock()
	if r.Method == http.MethodDelete {
		s.deleteAnimation(w, animation)
		return
	}
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		http.Error(w, "GET, HEAD, or DELETE only", http.StatusMethodNotAllowed)
		return
	}
	name := "animation-" + id + ".glb"
	if r.URL.Query().Get("download") == "1" {
		w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=%q", name))
	} else {
		w.Header().Set("Content-Disposition", fmt.Sprintf("inline; filename=%q", name))
	}
	w.Header().Set("X-Animation-Prompt", url.QueryEscape(prompt))
	serveGLB(w, r, path, name, createdAt)
}

func (s *server) deleteModel(w http.ResponseWriter, model *modelAsset) {
	s.mu.Lock()
	for _, animation := range s.jobs {
		if animation.ModelID == model.ID && (animation.State == "queued" || animation.State == "running") {
			s.mu.Unlock()
			http.Error(w, "model has an active animation", http.StatusConflict)
			return
		}
	}
	if err := s.store.deleteModel(model); err != nil {
		s.mu.Unlock()
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	delete(s.models, model.ID)
	for id, animation := range s.jobs {
		if animation.ModelID == model.ID {
			delete(s.jobs, id)
		}
	}
	s.mu.Unlock()
	w.WriteHeader(http.StatusNoContent)
}

func (s *server) deleteAnimation(w http.ResponseWriter, animation *animationRecord) {
	s.mu.Lock()
	model := s.models[animation.ModelID]
	if model == nil {
		s.mu.Unlock()
		http.Error(w, "not found", http.StatusNotFound)
		return
	}
	if err := s.store.deleteAnimation(model, animation); err != nil {
		s.mu.Unlock()
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	for index, candidate := range model.Animations {
		if candidate.ID == animation.ID {
			model.Animations = append(model.Animations[:index], model.Animations[index+1:]...)
			break
		}
	}
	delete(s.jobs, animation.ID)
	s.mu.Unlock()
	w.WriteHeader(http.StatusNoContent)
}

func serveGLB(w http.ResponseWriter, r *http.Request, path, name string, createdAt int64) {
	file, err := os.Open(path)
	if err != nil {
		http.NotFound(w, r)
		return
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || !info.Mode().IsRegular() {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "model/gltf-binary")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	modified := info.ModTime()
	if createdAt > 0 {
		modified = time.UnixMilli(createdAt)
	}
	http.ServeContent(w, r, name, modified, file)
}

func writeJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	if err := json.NewEncoder(w).Encode(value); err != nil {
		log.Printf("write JSON: %v", err)
	}
}

func (s *server) handleInfo(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "GET only", http.StatusMethodNotAllowed)
		return
	}
	writeJSON(w, map[string]any{
		"device": s.device, "model": s.model,
		"maxUploadMiB": s.store.maxUpload >> 20,
		"defaults": map[string]any{
			"seed": 666, "steps": 64, "guidance": 3,
			"trajectories": 0, "fps": 10, "maxVertices": 10000,
		},
	})
}

func sameOrigin(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Frame-Options", "DENY")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("Referrer-Policy", "no-referrer")
		w.Header().Set("Content-Security-Policy",
			"default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
		if r.Method != http.MethodGet && r.Method != http.MethodHead && r.Method != http.MethodOptions {
			if origin := r.Header.Get("Origin"); origin != "" {
				parsed, err := url.Parse(origin)
				if err != nil || !strings.EqualFold(parsed.Host, r.Host) {
					http.Error(w, "cross-origin mutation rejected", http.StatusForbidden)
					return
				}
			}
		}
		next.ServeHTTP(w, r)
	})
}

func regularFile(path, description string) error {
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("%s: %w", description, err)
	}
	if !info.Mode().IsRegular() {
		return fmt.Errorf("%s is not a regular file", description)
	}
	return nil
}

func main() {
	cliPath := flag.String("cli", "./result/bin/aam-cli", "path to aam-cli")
	modelPath := flag.String("model", "", "AnimateAnyMesh GGUF model (required)")
	storePath := flag.String("store", "./demo-data", "durable model and animation gallery")
	address := flag.String("addr", "127.0.0.1:8743", "listen address")
	device := flag.String("device", "auto", "aam-cli device: auto, cpu, or vulkan")
	threads := flag.Uint("threads", 0, "CPU inference threads (0 = automatic)")
	timeout := flag.Duration("timeout", 30*time.Minute, "maximum duration of one animation")
	maxUploadMiB := flag.Int64("max-upload-mib", defaultUploadLimit>>20, "maximum uploaded GLB size")
	flag.Parse()

	if *modelPath == "" {
		log.Fatal("-model is required")
	}
	if *device != "auto" && *device != "cpu" && *device != "vulkan" {
		log.Fatal("-device must be auto, cpu, or vulkan")
	}
	if *timeout <= 0 || *maxUploadMiB < 1 || *maxUploadMiB > 4096 {
		log.Fatal("-timeout and -max-upload-mib are outside their safe ranges")
	}
	if err := regularFile(*cliPath, "aam-cli"); err != nil {
		log.Fatal(err)
	}
	if err := regularFile(*modelPath, "GGUF model"); err != nil {
		log.Fatal(err)
	}
	store, err := openStore(*storePath, *maxUploadMiB<<20)
	if err != nil {
		log.Fatal(err)
	}
	runner := &cliAnimator{
		cli: *cliPath, model: *modelPath, device: *device,
		threads: *threads,
	}
	s, err := newServer(store, runner, *timeout, *device, *modelPath)
	if err != nil {
		log.Fatal(err)
	}
	go s.worker()

	web, err := fs.Sub(embeddedWeb, "web")
	if err != nil {
		log.Fatal(err)
	}
	mux := http.NewServeMux()
	logo, err := base64.StdEncoding.DecodeString(strings.TrimSpace(localAIBase64))
	if err != nil {
		log.Fatal("decode embedded LocalAI logo: ", err)
	}
	mux.HandleFunc("/localai.png", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			http.Error(w, "GET or HEAD only", http.StatusMethodNotAllowed)
			return
		}
		w.Header().Set("Content-Type", "image/png")
		w.Header().Set("Cache-Control", "public, max-age=86400")
		w.Header().Set("Content-Length", strconv.Itoa(len(logo)))
		if r.Method == http.MethodGet {
			_, _ = w.Write(logo)
		}
	})
	mux.HandleFunc("/api/info", s.handleInfo)
	mux.HandleFunc("/api/models", s.handleModels)
	mux.HandleFunc("/api/model/", s.handleModel)
	mux.HandleFunc("/api/job/", s.handleJob)
	mux.HandleFunc("/api/animation/", s.handleAnimation)
	mux.Handle("/", http.FileServer(http.FS(web)))

	log.Printf("restored %d model(s); listening on %s with %s", len(s.models), *address, *device)
	server := &http.Server{
		Addr: *address, Handler: sameOrigin(mux),
		ReadHeaderTimeout: 10 * time.Second, IdleTimeout: 2 * time.Minute,
		MaxHeaderBytes: 1 << 20,
	}
	log.Fatal(server.ListenAndServe())
}
