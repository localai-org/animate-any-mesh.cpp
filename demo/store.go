package main

import (
	"bufio"
	"crypto/rand"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
	"unicode/utf8"
)

const storeVersion = 1

type modelAsset struct {
	ID         string             `json:"id"`
	Name       string             `json:"name"`
	CreatedAt  int64              `json:"createdAt"`
	SizeBytes  int64              `json:"sizeBytes"`
	Thumbnail  string             `json:"thumbnail,omitempty"`
	Animations []*animationRecord `json:"-"`
	SourcePath string             `json:"-"`
	Directory  string             `json:"-"`
}

type animationOptions struct {
	Prompt       string  `json:"prompt"`
	Seed         uint64  `json:"seed"`
	Steps        int     `json:"steps"`
	Guidance     float64 `json:"guidance"`
	Trajectories int     `json:"trajectories"`
	FPS          float64 `json:"fps"`
	MaxVertices  int     `json:"maxVertices"`
}

type animationRecord struct {
	ID         string           `json:"id"`
	ModelID    string           `json:"modelId"`
	CreatedAt  int64            `json:"createdAt"`
	FinishedAt int64            `json:"finishedAt,omitempty"`
	DurationMS int64            `json:"durationMs,omitempty"`
	State      string           `json:"state"`
	Error      string           `json:"error,omitempty"`
	Backend    string           `json:"backend,omitempty"`
	Options    animationOptions `json:"options"`
	OutputPath string           `json:"-"`
	Directory  string           `json:"-"`
}

type modelManifest struct {
	Version   int    `json:"version"`
	ID        string `json:"id"`
	Name      string `json:"name"`
	CreatedAt int64  `json:"createdAt"`
	SizeBytes int64  `json:"sizeBytes"`
	Thumbnail string `json:"thumbnail,omitempty"`
}

type animationManifest struct {
	Version    int              `json:"version"`
	ID         string           `json:"id"`
	ModelID    string           `json:"modelId"`
	CreatedAt  int64            `json:"createdAt"`
	FinishedAt int64            `json:"finishedAt"`
	DurationMS int64            `json:"durationMs"`
	Backend    string           `json:"backend,omitempty"`
	Options    animationOptions `json:"options"`
}

type assetStore struct {
	root      string
	maxUpload int64
}

func openStore(root string, maxUpload int64) (*assetStore, error) {
	if root == "" {
		return nil, errors.New("store directory cannot be empty")
	}
	absolute, err := filepath.Abs(root)
	if err != nil {
		return nil, fmt.Errorf("resolve store directory: %w", err)
	}
	if err := os.MkdirAll(absolute, 0o755); err != nil {
		return nil, fmt.Errorf("create store directory: %w", err)
	}
	return &assetStore{root: absolute, maxUpload: maxUpload}, nil
}

func randomID() (string, error) {
	var value [8]byte
	if _, err := rand.Read(value[:]); err != nil {
		return "", fmt.Errorf("generate identifier: %w", err)
	}
	return hex.EncodeToString(value[:]), nil
}

func validID(value string) bool {
	if len(value) != 16 {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func cleanUploadName(value string) string {
	value = strings.TrimSpace(filepath.Base(value))
	if !utf8.ValidString(value) || value == "." || value == "" {
		return "model.glb"
	}
	runes := []rune(value)
	if len(runes) > 120 {
		value = string(runes[:120])
	}
	return value
}

func (s *assetStore) createModel(name string, source io.Reader) (*modelAsset, error) {
	id, err := randomID()
	if err != nil {
		return nil, err
	}
	temporary, err := os.MkdirTemp(s.root, ".model-"+id+"-")
	if err != nil {
		return nil, fmt.Errorf("create upload transaction: %w", err)
	}
	committed := false
	defer func() {
		if !committed {
			_ = os.RemoveAll(temporary)
		}
	}()

	sourcePath := filepath.Join(temporary, "source.glb")
	file, err := os.OpenFile(sourcePath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, fmt.Errorf("create uploaded GLB: %w", err)
	}
	written, copyErr := io.Copy(file, io.LimitReader(source, s.maxUpload+1))
	syncErr := file.Sync()
	closeErr := file.Close()
	if copyErr != nil {
		return nil, fmt.Errorf("store uploaded GLB: %w", copyErr)
	}
	if syncErr != nil {
		return nil, fmt.Errorf("sync uploaded GLB: %w", syncErr)
	}
	if closeErr != nil {
		return nil, fmt.Errorf("close uploaded GLB: %w", closeErr)
	}
	if written > s.maxUpload {
		return nil, fmt.Errorf("uploaded GLB exceeds the %d MiB limit", s.maxUpload>>20)
	}
	if err := validateGLBFile(sourcePath, s.maxUpload); err != nil {
		return nil, err
	}

	asset := &modelAsset{
		ID: id, Name: cleanUploadName(name), CreatedAt: time.Now().UnixMilli(),
		SizeBytes: written,
	}
	manifest := modelManifest{
		Version: storeVersion, ID: asset.ID, Name: asset.Name,
		CreatedAt: asset.CreatedAt, SizeBytes: asset.SizeBytes,
	}
	if err := writeJSONExclusive(filepath.Join(temporary, "manifest.json"), manifest); err != nil {
		return nil, err
	}
	finalDirectory := filepath.Join(s.root, id)
	if err := os.Rename(temporary, finalDirectory); err != nil {
		return nil, fmt.Errorf("commit uploaded model: %w", err)
	}
	committed = true
	asset.Directory = finalDirectory
	asset.SourcePath = filepath.Join(finalDirectory, "source.glb")
	return asset, nil
}

func (s *assetStore) restore() ([]*modelAsset, error) {
	entries, err := os.ReadDir(s.root)
	if err != nil {
		return nil, fmt.Errorf("read store: %w", err)
	}
	models := make([]*modelAsset, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() || !validID(entry.Name()) {
			continue
		}
		model, err := s.loadModel(filepath.Join(s.root, entry.Name()))
		if err != nil {
			continue
		}
		models = append(models, model)
	}
	sort.Slice(models, func(i, j int) bool { return models[i].CreatedAt > models[j].CreatedAt })
	return models, nil
}

func (s *assetStore) loadModel(directory string) (*modelAsset, error) {
	var manifest modelManifest
	if err := readJSON(filepath.Join(directory, "manifest.json"), &manifest); err != nil {
		return nil, err
	}
	if manifest.Version != storeVersion || !validID(manifest.ID) ||
		manifest.ID != filepath.Base(directory) {
		return nil, errors.New("invalid model manifest")
	}
	sourcePath := filepath.Join(directory, "source.glb")
	if err := validateGLBFile(sourcePath, s.maxUpload); err != nil {
		return nil, err
	}
	info, err := os.Stat(sourcePath)
	if err != nil || !info.Mode().IsRegular() {
		return nil, errors.New("model source is not a regular file")
	}
	model := &modelAsset{
		ID: manifest.ID, Name: manifest.Name, CreatedAt: manifest.CreatedAt,
		SizeBytes: info.Size(), Thumbnail: manifest.Thumbnail,
		SourcePath: sourcePath, Directory: directory,
	}
	animationDirectory := filepath.Join(directory, "animations")
	entries, err := os.ReadDir(animationDirectory)
	if err != nil {
		if os.IsNotExist(err) {
			return model, nil
		}
		return nil, err
	}
	for _, entry := range entries {
		if !entry.IsDir() || !validID(entry.Name()) {
			continue
		}
		dir := filepath.Join(animationDirectory, entry.Name())
		var saved animationManifest
		if err := readJSON(filepath.Join(dir, "manifest.json"), &saved); err != nil ||
			saved.Version != storeVersion || saved.ID != entry.Name() || saved.ModelID != model.ID {
			continue
		}
		outputPath := filepath.Join(dir, "animation.glb")
		if err := validateGLBFile(outputPath, 1<<32-1); err != nil {
			continue
		}
		model.Animations = append(model.Animations, &animationRecord{
			ID: saved.ID, ModelID: saved.ModelID, CreatedAt: saved.CreatedAt,
			FinishedAt: saved.FinishedAt, DurationMS: saved.DurationMS,
			State: "done", Backend: saved.Backend, Options: saved.Options,
			OutputPath: outputPath, Directory: dir,
		})
	}
	sort.Slice(model.Animations, func(i, j int) bool {
		return model.Animations[i].CreatedAt > model.Animations[j].CreatedAt
	})
	return model, nil
}

func (s *assetStore) beginAnimation(model *modelAsset, animation *animationRecord) (string, string, error) {
	root := filepath.Join(model.Directory, "animations")
	if err := os.MkdirAll(root, 0o755); err != nil {
		return "", "", fmt.Errorf("create animations directory: %w", err)
	}
	temporary, err := os.MkdirTemp(root, ".animation-"+animation.ID+"-")
	if err != nil {
		return "", "", fmt.Errorf("create animation transaction: %w", err)
	}
	return temporary, filepath.Join(temporary, "animation.glb"), nil
}

func (s *assetStore) commitAnimation(model *modelAsset, animation *animationRecord, temporary string) error {
	output := filepath.Join(temporary, "animation.glb")
	if err := validateGLBFile(output, 1<<32-1); err != nil {
		return fmt.Errorf("generated output: %w", err)
	}
	manifest := animationManifest{
		Version: storeVersion, ID: animation.ID, ModelID: animation.ModelID,
		CreatedAt: animation.CreatedAt, FinishedAt: animation.FinishedAt,
		DurationMS: animation.DurationMS, Backend: animation.Backend,
		Options: animation.Options,
	}
	if err := writeJSONExclusive(filepath.Join(temporary, "manifest.json"), manifest); err != nil {
		return err
	}
	finalDirectory := filepath.Join(model.Directory, "animations", animation.ID)
	if err := os.Rename(temporary, finalDirectory); err != nil {
		return fmt.Errorf("commit animation: %w", err)
	}
	animation.Directory = finalDirectory
	animation.OutputPath = filepath.Join(finalDirectory, "animation.glb")
	return nil
}

func (s *assetStore) saveThumbnail(model *modelAsset, thumbnail string) error {
	manifest := modelManifest{
		Version: storeVersion, ID: model.ID, Name: model.Name,
		CreatedAt: model.CreatedAt, SizeBytes: model.SizeBytes, Thumbnail: thumbnail,
	}
	if err := writeJSONAtomic(filepath.Join(model.Directory, "manifest.json"), manifest); err != nil {
		return err
	}
	model.Thumbnail = thumbnail
	return nil
}

func (s *assetStore) deleteModel(model *modelAsset) error {
	if model == nil || !validID(model.ID) || filepath.Dir(model.Directory) != s.root {
		return errors.New("refusing to delete an invalid model path")
	}
	return os.RemoveAll(model.Directory)
}

func (s *assetStore) deleteAnimation(model *modelAsset, animation *animationRecord) error {
	expectedParent := filepath.Join(model.Directory, "animations")
	if animation == nil || !validID(animation.ID) || filepath.Dir(animation.Directory) != expectedParent {
		return errors.New("refusing to delete an invalid animation path")
	}
	return os.RemoveAll(animation.Directory)
}

func validateGLBFile(path string, maximum int64) error {
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open GLB: %w", err)
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || !info.Mode().IsRegular() {
		return errors.New("GLB is not a regular file")
	}
	if info.Size() < 20 || info.Size() > maximum {
		return fmt.Errorf("GLB size %d is outside the allowed range", info.Size())
	}
	var header [20]byte
	if _, err := io.ReadFull(file, header[:]); err != nil {
		return fmt.Errorf("read GLB header: %w", err)
	}
	if binary.LittleEndian.Uint32(header[0:4]) != 0x46546c67 ||
		binary.LittleEndian.Uint32(header[4:8]) != 2 ||
		int64(binary.LittleEndian.Uint32(header[8:12])) != info.Size() {
		return errors.New("invalid GLB header, version, or declared length")
	}
	jsonLength := int64(binary.LittleEndian.Uint32(header[12:16]))
	if binary.LittleEndian.Uint32(header[16:20]) != 0x4e4f534a ||
		jsonLength < 2 || jsonLength > info.Size()-20 {
		return errors.New("GLB does not begin with a valid JSON chunk")
	}
	return nil
}

func readJSON(path string, output any) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	decoder := json.NewDecoder(bufio.NewReader(io.LimitReader(file, 2<<20)))
	decoder.DisallowUnknownFields()
	return decoder.Decode(output)
}

func writeJSONExclusive(path string, value any) error {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(value); err != nil {
		file.Close()
		return err
	}
	if err := file.Sync(); err != nil {
		file.Close()
		return err
	}
	return file.Close()
}

func writeJSONAtomic(path string, value any) error {
	temporary, err := os.CreateTemp(filepath.Dir(path), ".manifest-")
	if err != nil {
		return err
	}
	temporaryPath := temporary.Name()
	committed := false
	defer func() {
		if !committed {
			_ = os.Remove(temporaryPath)
		}
	}()
	if err := temporary.Chmod(0o644); err != nil {
		temporary.Close()
		return err
	}
	encoder := json.NewEncoder(temporary)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(value); err != nil {
		temporary.Close()
		return err
	}
	if err := temporary.Sync(); err != nil {
		temporary.Close()
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	if err := os.Rename(temporaryPath, path); err != nil {
		return err
	}
	committed = true
	return nil
}
