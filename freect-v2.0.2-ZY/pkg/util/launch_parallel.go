package util

import (
	"runtime"
	"sync"
)

// MAX_PROC specifies how many processes are allowed when using
// LaunchParallel* functions. Positive numbers specify the number of
// processes.  Negative numbers specify fractions of the total number
// of cores.  E.g., if 8 cores are available, MAX_PROC=-1 will use all
// 8; MAX_PROC=-2 will use 4; etc.  Default of -2 is chosen since most
// processors enable hyperthreading, and there is little to gain using
// those extra threads.
var MAX_PROC = -2

// LaunchParallel distributes work in operands accross num_goroutines
// goroutines.  Ideal for operating on arrays where each index is independent.
func LaunchParallel(worker func([]int, *sync.WaitGroup), operands []int, num_goroutines int) {

	if num_goroutines < 0 {
		num_goroutines = runtime.NumCPU() / (-num_goroutines)
	}

	// Initialize slice of integer slices
	workloads := make([]([]int), num_goroutines)
	for i := range workloads {
		workloads[i] = make([]int, 0)
	}

	for i, v := range operands {
		curr_goroutine := i % num_goroutines
		workloads[curr_goroutine] = append(workloads[curr_goroutine], v)
	}

	wg := &sync.WaitGroup{}
	for i := 0; i < num_goroutines; i++ {
		wg.Add(1)
		go worker(workloads[i], wg)
	}

	wg.Wait()
}

// Convenience t
func LaunchParallelN(worker func([]int, *sync.WaitGroup), N int, num_goroutines int) {
	operands := make([]int, N)
	for i := range operands {
		operands[i] = i
	}
	LaunchParallel(worker, operands, num_goroutines)
}
