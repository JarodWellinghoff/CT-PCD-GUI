package main

import (

	//"flag"
	//log "github.com/sirupsen/logrus"
	//"gitlab.com/freect/freect/pkg/raw"
	//"github.com/briandowns/spinner"
	//"time"
	"github.com/spf13/cobra"
)

type Spinner interface {
	Start()
	Stop()
}

type emptySpinner struct{}

func (e *emptySpinner) Start() {}
func (e *emptySpinner) Stop()  {}

var rootCmd = &cobra.Command{
	Use:   "fct-raw-util",
	Short: "fct-raw-util performs tasks associated with raw data",
	Long:  `CT raw data is just the worst.  This tool hopefully makes it a little better.`,
}

func main() {

	//input_dir := ""
	//output_dir := ""
	//progress_indicator := true
	//
	//flag.StringVar(&input_dir, "i", input_dir, "Source/Input directory for data conversion (required)")
	//flag.StringVar(&output_dir, "o", output_dir, "Output directory for data conversion (required)")
	//flag.BoolVar(&progress_indicator, "progress", progress_indicator, "Show progress indicator (default: true)")
	//flag.Parse()
	//
	//if input_dir == "" || output_dir == "" {
	//	log.Error("Missing required arguments")
	//	flag.Usage()
	//}
	//
	//log.Infof("freect raw data converstion requested\n    input: %s\n    output: %s", input_dir, output_dir)
	//
	//var s Spinner
	//if progress_indicator {
	//	s = spinner.New(spinner.CharSets[14], 100*time.Millisecond)
	//} else {
	//	s = &emptySpinner{}
	//}
	//
	//// Load raw data into memory
	//log.Info("converting raw data...")
	//s.Start()
	////r := raw.NewRawDataReader(raw.FromDicom(input_dir))
	//
	//d_in := raw.NewLocalDirectoryStore(input_dir)
	//r, err := raw.NewBattelleReader(d_in)
	//if err!= nil {
	//	log.Fatal(err)
	//}
	//
	//d_out := raw.NewLocalDirectoryStore(output_dir)
	//err = raw.NewBinaryWriter(d_out).Write(r)
	//if err != nil {
	//	log.Fatal(err)
	//}
	//
	//s.Stop()
	//
	//log.Info("raw data save complete")

	rootCmd.Execute()
}
