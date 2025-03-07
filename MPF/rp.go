package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
)

// pythonRunner represents a Python script runner
type pythonRunner struct {
	filename string
	cmd      *exec.Cmd
	wg       *sync.WaitGroup
}

// newPythonRunner creates a new pythonRunner instance
func newPythonRunner(filename string, wg *sync.WaitGroup) *pythonRunner {
	return &pythonRunner{
		filename: filename,
		wg:       wg,
	}
}

// run executes the Python script and handles its output
func (pr *pythonRunner) run() {
	defer pr.wg.Done()

	// Get the current directory
	currentDir, err := os.Getwd()
	if err != nil {
		log.Printf("Error getting current directory: %v\n", err)
		return
	}

	// Construct full path to the Python script
	scriptPath := filepath.Join(currentDir, pr.filename)

	// Check if the file exists
	if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
		log.Printf("Error: File not found: %s\n", scriptPath)
		return
	}

	// Create the command
	pr.cmd = exec.Command("python", scriptPath)

	// Create pipes for stdout and stderr
	stdout, err := pr.cmd.StdoutPipe()
	if err != nil {
		log.Printf("Error creating stdout pipe for %s: %v\n", pr.filename, err)
		return
	}

	stderr, err := pr.cmd.StderrPipe()
	if err != nil {
		log.Printf("Error creating stderr pipe for %s: %v\n", pr.filename, err)
		return
	}

	// Start the command
	if err := pr.cmd.Start(); err != nil {
		log.Printf("Error starting %s: %v\n", pr.filename, err)
		return
	}

	// Create a WaitGroup for the output handlers
	var outputWg sync.WaitGroup
	outputWg.Add(2)

	// Handle stdout in a goroutine
	go func() {
		defer outputWg.Done()
		buffer := make([]byte, 1024)
		for {
			n, err := stdout.Read(buffer)
			if n > 0 {
				fmt.Printf("[%s] %s", pr.filename, buffer[:n])
			}
			if err != nil {
				break
			}
		}
	}()

	// Handle stderr in a goroutine
	go func() {
		defer outputWg.Done()
		buffer := make([]byte, 1024)
		for {
			n, err := stderr.Read(buffer)
			if n > 0 {
				fmt.Printf("[%s][ERROR] %s", pr.filename, buffer[:n])
			}
			if err != nil {
				break
			}
		}
	}()

	// Wait for the command to complete
	err = pr.cmd.Wait()

	// Wait for output handlers to finish
	outputWg.Wait()

	if err != nil {
		log.Printf("%s completed with error: %v\n", pr.filename, err)
	} else {
		log.Printf("%s completed successfully\n", pr.filename)
	}
}

func main() {
	// List of Python files to run
	pythonFiles := []string{
		"Gesture_Controller.py",
		"proton.py",
	}

	// Create a WaitGroup to wait for all scripts to complete
	var wg sync.WaitGroup
	wg.Add(len(pythonFiles))

	// Create and start a runner for each Python file
	runners := make([]*pythonRunner, len(pythonFiles))
	for i, file := range pythonFiles {
		runner := newPythonRunner(file, &wg)
		runners[i] = runner
		go runner.run()
	}

	// Wait for all scripts to complete
	wg.Wait()

	fmt.Println("All Python scripts have finished executing")
}
