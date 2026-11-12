package shared_cmd

import (
	"fmt"
	log "github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"gitlab.com/freect/freect/pkg/image"
	"gitlab.com/freect/freect/pkg/raw"
	"gitlab.com/freect/freect/pkg/recon"
	"gitlab.com/freect/freect/pkg/util"
	"gopkg.in/yaml.v3"
	"os"
	"reflect"
	"strings"
	"time"
)

func FreeCTLogo() {
	fmt.Println("================================================================================")
	fmt.Println("")
	fmt.Println(` ______              _____ _______ `)
	fmt.Println(`|  ____|            / ____|__   __|`)
	fmt.Println(`| |__ _ __ ___  ___| |       | |   `)
	fmt.Println(`|  __| '__/ _ \/ _ \ |       | |   `)
	fmt.Println(`| |  | | |  __/  __/ |____   | |   `)
	fmt.Println(`|_|  |_|  \___|\___|\_____|  |_|   `)
	fmt.Println("")
	fmt.Println(util.Version())
	fmt.Println("")
	fmt.Println("================================================================================")
}

func ReconWarning() {
	fmt.Println("================================================================================")
	fmt.Println("")
	fmt.Println("FreeCT is free software made available for research under a modified BSD license")
	fmt.Println("Please visit https://gitlab.com/freect/freect to review these terms")
	fmt.Println("")
	fmt.Println("NOT FOR CLINICAL USE")
	fmt.Println("")
	fmt.Println("FreeCT uses very minimal telemetry to get a count of active users.  This helps ")
	fmt.Println("helps us understand demand for our tools, and apply for funding to continue to")
	fmt.Println("support FreeCT development.  No identifiable data is stored.  We'd appreciate")
	fmt.Println("if you leave this enabled, however we understand if you wish to disable it.")
	fmt.Println("To disable, simple run your command with the \"--no-telemetry\" argument.")
	fmt.Println("")
	fmt.Println("================================================================================")
}

func PrintConfig(c recon.ReconConfig) {
	fmt.Println("================================================================================")
	fmt.Println("Final reconstruction config (including command line overrides):")
	fmt.Println("")
	data, err := yaml.Marshal(c)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(string(data))
	fmt.Println("================================================================================")
}

func PrintFinalMessage() {
	fmt.Println("================================================================================")
	fmt.Println("Thank you for using FreeCT!")
	fmt.Println("Feedback (good and bad), bugs, feature requests, etc. can be submitted at")
	fmt.Println("")
	fmt.Println("        https://gitlab.com/freect/freect/-/issues")
	fmt.Println("")
	fmt.Println("================================================================================")
}

func GenerateReconCommand(c *recon.ReconConfig, rootCmd *cobra.Command, reconFunc recon.ReconstructionFunc) {
	var cmdRecon = &cobra.Command{
		Use:   "recon [flags] config_file",
		Short: "Execute a reconstruction",
		Long:  "",
		Args:  cobra.MinimumNArgs(1),
		PreRun: func(cmd *cobra.Command, args []string) {

		},

		Run: func(cmd *cobra.Command, args []string) {

			start := time.Now()
			
			// Grab the original structure since this is what will
			// hold any override information
			rs_org_struct := c.ReconSpecific

			// Load the configuration file
			read, err := os.Open(args[0])
			if err != nil {
				log.Fatal(err)
			}

			err = yaml.NewDecoder(read).Decode(c)
			if err != nil {
				log.Fatal(err)
			}

			// KLUDGE: Unfortunately, my brilliant plan for config
			// handling of different types of reconstruction
			// configuration is a no-go since the yaml decoder doesn't
			// unpack the ReconSpecific struct into the underlying
			// type, but rather into a generic map[string]any.  I
			// sorta shoulda seen that coming.  For the time being,
			// since WFBP is the only one we're supporting, I'm going
			// to plow ahead with the current setup (still using the
			// interface) but just manually reset the structure in the
			// reconFunc.  This SHOULD work 100% of the time, since
			// the base config will have been generated from the original
			// structure.
			//
			// TODO: Revisit the config structure design and see if I
			// can find a better way to handle this.

			// Parse flags to override the config that we loaded.
			// This is somewhat inefficient, since cobra has already parsed them once
			// and also could cause issues some day since I'm not respecting the command
			// tree, but for right now I just need to get it running, and this works.
			err = cmd.ParseFlags(os.Args[1:])
			if err != nil {
				log.Fatal(err)
			}

			if !recon.GlobalRuntimeConfig.Silent {
				FreeCTLogo()
				ReconWarning()
				time.Sleep(500*time.Millisecond)
				PrintConfig(*c)
			}

			images, err := reconFunc(c, rs_org_struct)
			if err != nil {
				log.Fatal(err)
			}

			// Save images based on requested output
			log.Info("Saving images...")
			
			if c.ImageOutput.DICOM != "" {
				ld := raw.NewLocalDirectoryStore(c.ImageOutput.DICOM)
				dw := image.NewDICOMWriter(ld, "")

				log.Info("Writer info: ", dw.String())

				err := dw.Write(image.ImageStack(images))
				if err != nil {
					log.Error("DICOM output failed:", err)
				}
			}

			if c.ImageOutput.NIfTI1 != "" {
				f, err := os.Create(c.ImageOutput.NIfTI1)
				if err != nil {
					log.Error("NIfTI output failed:", err)
				}

				nw := image.NewNifti1Writer(f)
				log.Info("Writer info: ", nw.String(), " saving to ", c.ImageOutput.NIfTI1)
				
				err = nw.Write(image.ImageStack(images))
				if err != nil {
					log.Error("NIfTI output failed:", err)
				}
			}

			log.Info("Reconstruction complete!")
			log.Infof("Recontructed %d slices in %v", len(images), time.Since(start))

			PrintFinalMessage()
		},
	}

	ReconGenerateFlags(c, cmdRecon)

	cmdRecon.Flags().BoolVar(&recon.GlobalRuntimeConfig.Silent, "silent", recon.GlobalRuntimeConfig.Silent, "run recon silently (default: false)")

	rootCmd.AddCommand(cmdRecon)
}

func ReconGenerateFlags(in *recon.ReconConfig, cmd *cobra.Command) {

	in_type := reflect.TypeOf(*in)
	in_val := reflect.ValueOf(*in)

	for _, f := range reflect.VisibleFields(in_type) {

		group_name := f.Name

		field_val := in_val.FieldByName(f.Name).Elem() // Return the concrete substructures pointed to by "in"

		if field_val.Kind() != reflect.Struct {
			field_val = field_val.Elem()
		}

		sub_fields := reflect.VisibleFields(field_val.Type())
		for _, sub_f := range sub_fields {
			tag_name := sub_f.Name
			sub_v := field_val.FieldByName(tag_name)
			full_flag_name := strings.ToLower(group_name) + "." + strings.ToLower(tag_name)

			switch sub_v.Type() {
			case reflect.TypeOf(string("")):
				p := sub_v.Addr().Interface().(*string)
				cmd.Flags().StringVar(p, full_flag_name, sub_v.String(), "")
			case reflect.TypeOf(int(0)):
				p := sub_v.Addr().Interface().(*int)
				cmd.Flags().IntVar(p, full_flag_name, int(sub_v.Int()), "")
			case reflect.TypeOf(float64(0.0)):
				p := sub_v.Addr().Interface().(*float64)
				cmd.Flags().Float64Var(p, full_flag_name, sub_v.Float(), "")
			case reflect.TypeOf(bool(false)):
				p := sub_v.Addr().Interface().(*bool)
				cmd.Flags().BoolVar(p, full_flag_name, sub_v.Bool(), "")
			}
		}
	}
}
