package util

import (
	"testing"
)

func TestLogUpdate(t *testing.T) {
	N := 10000
	for i := 0; i < N; i++ {
		LogUpdate(i, N, 10, "long running process ")
	}
}
