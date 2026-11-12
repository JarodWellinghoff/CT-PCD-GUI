/*
Copyright © 2022 NAME HERE <EMAIL ADDRESS>

*/
package cmd

import (
	"encoding/xml"
	"fmt"
	log "github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"hash/fnv"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// Helper function
func hash(s string) string {
	h := fnv.New32a()
	h.Write([]byte(s))
	return fmt.Sprintf("%x", h.Sum32())
}

// This dedicated type is required to serialize SliceWidth for ReconCT into two decimal places
type TwoDecimalFloat float64

func (f TwoDecimalFloat) MarshalXML(e *xml.Encoder, start xml.StartElement) error {
	s := fmt.Sprintf("%0.2f", f)
	return e.EncodeElement(s, start)
}

type ReconJobViewModel struct {
	ReconMode            string           `xml:"ReconMode,omitempty"`
	ConvolutionKernel    string           `xml:"ConvolutionKernel,omitempty"`
	SliceWidth           TwoDecimalFloat  `xml:"SliceWidth,omitempty"`
	SeriesDescription    string           `xml:"SeriesDescription,omitempty"`
	FieldOfView          float64          `xml:"FieldOfView,omitempty"`
	AnonymizeDicomImages bool             `xml:"AnonymizeDicomImages,omitempty"`
	PatientName          string           `xml:"PatientName,omitempty"`
	PatientId            string           `xml:"PatientId,omitempty"`
	NoiseSimulation      *NoiseSimulation `xml:"NoiseSimulation,omitempty"`
}

type NoiseSimulation struct {
	NoiseSimulationChecked         bool   `xml:"NoiseSimulationChecked"`
	NoiseSimulationItems           string `xml:"NoiseSimulationItems"`
	NoiseSimulationReductionFactor int    `xml:"NoiseSimulationReductionFactor"`
}

type CLISettings struct {
	ptrFile        string
	useWFBP        bool
	useSAFIRE      bool
	useADMIRE      bool
	sliceThickness string
	dose           string
	kernel         string
	output         string
	fov            string
	anonymize      bool
	imageOutput    string
}

var s = CLISettings{
	dose:        "100",
	fov:         "275",
	anonymize:   true,
	imageOutput: "",
}

// generateReconctXmlCmd represents the generateReconctXml command
var generateReconctXmlCmd = &cobra.Command{
	Use:   "generate-reconct-xml [flags] raw_data_file.ptr",
	Short: "Generate an input file for ReconCT",
	Args:  cobra.ExactArgs(1), // We may want to expand this to MinimumArgs(n int) in the future to inputs for multiple ptr files as well
	Run: func(cmd *cobra.Command, args []string) {

		ptr_file := args[0]

		//ptr_dirpath, ptr_filename := filepath.Split(ptr_file)
		_, ptr_filename := filepath.Split(ptr_file)

		// Get individual slice thicknesses
		// TODO this will fail if you don't specify slice thickness on command line.  Fix that.
		sts_strings := strings.Split(s.sliceThickness, ",")
		sts := make([]float64, len(sts_strings))
		for i, s := range sts_strings {
			st, err := strconv.ParseFloat(s, 64)
			if err != nil {
				log.Fatal(err)
			}
			sts[i] = st
		}

		// Get individual doses
		// TODO this will fail if you don't specify slice thickness on command line.  Fix that.
		dose_strings := strings.Split(s.dose, ",")
		doses := make([]float64, len(dose_strings))
		for i, s := range dose_strings {
			st, err := strconv.ParseFloat(s, 64)
			if err != nil {
				log.Fatal(err)
			}
			doses[i] = st
		}

		// Get Kernels
		kernels := strings.Split(s.kernel, ",")

		algs := make([]string, 0)
		if s.useWFBP {
			algs = append(algs, "Standard - WFBP")
		}

		if s.useSAFIRE {
			algs = append(algs, "Standard - SAFIRE")
		}

		if s.useADMIRE {
			algs = append(algs, "Standard - ADMIRE")
		}

		// Get FOV (only one supported a.t.m.)
		fov, err := strconv.ParseFloat(s.fov, 64)
		if err != nil {
			log.Fatal(err)
		}

		// Because I haven't implemented my new object store/filesystem

		// handle any lingering anonymization
		// This is a bit confusing see commites
		patient_name := "" // these will be OMITTED, causing ReconCT to inject the original values into the job
		patient_id := ""   // these will be OMITTED, causing ReconCT to inject the original values into the job
		if s.anonymize {
			patient_name = hash(ptr_file) // This will NOT be ommitted, and will override the original values
			patient_id = hash(ptr_file)   // This will NOT be ommitted, and will override the original values
		}

		// Generate the input structs
		for _, alg := range algs {
			for _, dose := range doses {
				for _, kernel := range kernels {
					for _, st := range sts {

						var noise_simulation *NoiseSimulation = nil
						if dose != 100.0 {
							noise_simulation = &NoiseSimulation{
								NoiseSimulationChecked:         true,
								NoiseSimulationItems:           "ExFactor",
								NoiseSimulationReductionFactor: int(dose),
							}
						}

						r := ReconJobViewModel{
							ReconMode:            alg,
							ConvolutionKernel:    kernel,
							SliceWidth:           TwoDecimalFloat(st),
							FieldOfView:          fov,
							AnonymizeDicomImages: s.anonymize,
							PatientName:          patient_name,
							PatientId:            patient_id,
							NoiseSimulation:      noise_simulation,
						}

						config_name := fmt.Sprintf("%s-%0.1f-%s-%s-%0.2f.xml", ptr_filename, dose, alg, kernel, st)

						var config_writer io.Writer
						var shell_writer io.Writer
						if s.output == "stdout" {
							config_writer = os.Stdout
							shell_writer = os.Stdout
						} else {
							config_path := filepath.Join(s.output, config_name)

							f_config, err := os.Create(config_path)
							if err != nil {
								log.Fatal(err)
							}

							config_writer = f_config

							shell_path := filepath.Join(s.output, config_name+".sh")
							f_shell, err := os.Create(shell_path)
							if err != nil {
								log.Fatal(err)
							}

							shell_writer = f_shell

						}

						image_output := filepath.Join(s.imageOutput, fmt.Sprintf("%.2f-%s-%.2f", dose, kernel, st))

						writeConfig(config_writer, r, config_name)
						writeShell(shell_writer, ptr_file, config_name, image_output)
					}
				}
			}
		}
	},
}

func writeConfig(wr io.Writer, r ReconJobViewModel, config_name string) {

	fmt.Println("--------------------------------------------------")
	fmt.Println(config_name)
	fmt.Println("--------------------------------------------------")

	data, err := xml.MarshalIndent(r, "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Fprintf(wr, string(data)+"\n")
}

func writeShell(wr io.Writer, ptr string, config_name string, image_output string) {

	if wr == os.Stdout {
		fmt.Println("")
		fmt.Println("Shell command to invoke:")
		fmt.Println("")
		fmt.Printf("    ")
		defer fmt.Println("")
	}

	if image_output != "" {
		fmt.Fprintf(wr, "/d/ReconCT_14.2.0.40998/ReconCT.exe -s -r:\"%s\" -p:\"%s\" -autoshutdown -setting:ImageDirectory=\"%s\"\n", ptr, config_name, image_output)
	} else {
		fmt.Fprintf(wr, "/d/ReconCT_14.2.0.40998/ReconCT.exe -s -r:\"%s\" -p:\"%s\" -autoshutdown\n", ptr, config_name)
	}

}

func init() {
	rootCmd.AddCommand(generateReconctXmlCmd)

	// Because this tool is highly specialized, we manually generate the required flags
	generateReconctXmlCmd.Flags().BoolVar(&s.useWFBP, "wfbp", s.useWFBP, "Use WFBP reconstruction algorihtm")
	generateReconctXmlCmd.Flags().BoolVar(&s.useSAFIRE, "safire", s.useSAFIRE, "Use SAFIRE reconstruction algorithm")
	generateReconctXmlCmd.Flags().BoolVar(&s.useADMIRE, "admire", s.useADMIRE, "Use ADMIRE reconstruction algorithm")

	generateReconctXmlCmd.Flags().StringVar(&s.imageOutput, "imageoutput", s.imageOutput, "Output directory for reconstructed images")

	generateReconctXmlCmd.Flags().BoolVar(&s.anonymize, "anonymize", s.anonymize, "Anonymize the output dicom images (default True)")

	generateReconctXmlCmd.Flags().StringVar(&s.sliceThickness, "slicethickness", s.sliceThickness, "Slice thickness(es) to reconstruction with")
	generateReconctXmlCmd.Flags().StringVar(&s.dose, "dose", s.dose, "Dose(s) to reconstruction with")
	generateReconctXmlCmd.Flags().StringVar(&s.kernel, "kernel", s.kernel, "Kernel(s) to reconstruction with. Note: make sure kernels are compatible with algorithm selection.")
	generateReconctXmlCmd.Flags().StringVar(&s.fov, "fov", s.fov, "Field of view to use for reconstructions (only supports a single value, defaults to 275mm)")

	generateReconctXmlCmd.Flags().StringVar(&s.output, "output", "stdout", "Where to save output. Default prints to stdout.")
}
