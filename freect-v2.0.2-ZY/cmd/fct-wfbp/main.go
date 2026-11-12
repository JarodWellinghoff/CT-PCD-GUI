package main

import (
	//log "github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"gitlab.com/freect/freect/internal/shared_cmd"
	"gitlab.com/freect/freect/pkg/recon"
	"gitlab.com/freect/freect/pkg/recon/wfbp"
	"gitlab.com/freect/freect/pkg/util"
	"os"
)

func init() {
	//log.SetLevel(log.DebugLevel)
}

var rootCmd = &cobra.Command{
	Use:   "fct-wfbp",
	Short: "fct-wfbp is GPU-enabled, clinical-quality reconstruction software",
	Long: `Create a weighed filtered backrpojection reconstruction
                from different raw data input types.`,
}

// See wfbp.WFBPReconstructionFunc for the actual implementation of WFBP reconstruction
func main() {

	rootCmd.PersistentFlags().BoolVar(&recon.GlobalRuntimeConfig.NoTelemetry, "no-telemetry", recon.GlobalRuntimeConfig.NoTelemetry, "disable telemetry packet")
	rootCmd.ParseFlags(os.Args)

	// NOTE: We don't check the error here because no arguments other
	// than no-telemetry are registered at this point, and we will log
	// errors that are not actually errors.  We need to parse it first
	// though to respect user choice.  There's probably a cleaner way
	// to handle all of this and at some point we should revisit all
	// of the argument setup to clean it up. TODO
	//
	//if err != nil { log.Error(err) }

	c := make(chan struct{})
	go util.FireTelemetry(recon.GlobalRuntimeConfig.NoTelemetry, c)

	var wfbp_conf = wfbp.DefaultWFBPConfig()
	var conf = recon.NewReconConfig(wfbp_conf)

	shared_cmd.AddVersionCommand(rootCmd)
	shared_cmd.GenerateEmptyConfigCommand(conf, rootCmd)
	shared_cmd.GenerateBaseConfigCommand(conf, rootCmd)

	// Note: we manually zero the fields here to preserve the
	// pointer. The wfbp_conf struct is what will capture the command
	// line overrides (if any).  Unfortunately at this time, we must
	// use the zero value to determine if these were set/not set.
	wfbp_conf.KernelA = 0.0
	wfbp_conf.KernelC = 0.0
	var recon_func recon.ReconstructionFunc = wfbp.WFBPReconstructionFunc
	shared_cmd.GenerateReconCommand(conf, rootCmd, recon_func)
	rootCmd.Execute()

	// Block until telemetry completes
	<-c
}
