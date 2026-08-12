package main

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"
)

type animator interface {
	Animate(context.Context, string, string, animationOptions) (string, error)
}

type cliAnimator struct {
	cli     string
	model   string
	device  string
	threads uint
}

type limitedLog struct {
	buffer bytes.Buffer
	limit  int
}

func (l *limitedLog) Write(value []byte) (int, error) {
	original := len(value)
	if l.buffer.Len() < l.limit {
		remaining := l.limit - l.buffer.Len()
		if len(value) > remaining {
			value = value[:remaining]
		}
		_, _ = l.buffer.Write(value)
	}
	return original, nil
}

func (l *limitedLog) String() string { return strings.TrimSpace(l.buffer.String()) }

func (a *cliAnimator) Animate(
	ctx context.Context, sourcePath, outputPath string, options animationOptions,
) (string, error) {
	arguments := []string{
		"animate", "--model", a.model, "--input", sourcePath, "--output", outputPath,
		"--prompt", options.Prompt, "--device", a.device,
		"--seed", fmt.Sprint(options.Seed), "--steps", fmt.Sprint(options.Steps),
		"--guidance", fmt.Sprint(options.Guidance), "--fps", fmt.Sprint(options.FPS),
		"--max-vertices", fmt.Sprint(options.MaxVertices),
	}
	if options.Trajectories != 0 {
		arguments = append(arguments, "--trajectories", fmt.Sprint(options.Trajectories))
	}
	if a.threads != 0 {
		arguments = append(arguments, "--threads", fmt.Sprint(a.threads))
	}
	command := exec.CommandContext(ctx, a.cli, arguments...)
	command.Dir = filepath.Dir(outputPath)
	logs := &limitedLog{limit: 64 << 10}
	command.Stdout = logs
	command.Stderr = logs
	if err := command.Run(); err != nil {
		if ctx.Err() != nil {
			return "", fmt.Errorf("animation timed out: %w", ctx.Err())
		}
		message := logs.String()
		if message == "" {
			message = err.Error()
		}
		return "", fmt.Errorf("aam-cli failed: %s", message)
	}
	backend := ""
	for _, line := range strings.Split(logs.String(), "\n") {
		if value, found := strings.CutPrefix(strings.TrimSpace(line), "backend:"); found {
			backend = strings.TrimSpace(value)
			break
		}
	}
	return backend, nil
}
