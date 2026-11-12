package util

import (
	"fmt"
	log "github.com/sirupsen/logrus"
)

// LogUpdate logs "message" with a percentage completion based
// inputs i and N
func LogUpdate(i int, N int, num_updates int, message string) {
	interval := N / num_updates
	if i%interval == 0 {
		perc_complete := int(float64(100) * float64(i) / float64(N))
		comp_update := fmt.Sprintf("(%d/%d %d%%)", i, N, perc_complete)
		log.Info("    ", message, " ", comp_update)
	} else if i == (N - 1) {
		log.Info("    ", message, " (100%)")
	}
}
