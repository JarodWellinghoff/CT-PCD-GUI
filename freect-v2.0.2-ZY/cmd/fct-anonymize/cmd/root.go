/*
Copyright © 2022 NAME HERE <EMAIL ADDRESS>

*/
package cmd

import (
	log "github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	dicom "gitlab.com/freect/go-dicom"
	"gitlab.com/freect/go-dicom/pkg/tag"
	//"github.com/suyashkumar/dicom"
	//"github.com/suyashkumar/dicom/pkg/tag"
	"os"
	"path/filepath"
)

// rootCmd represents the base command when called without any subcommands
var rootCmd = &cobra.Command{
	Use:   "fct-anonymize",
	Short: "A brief description of your application",
	Long: `A longer description that spans multiple lines and likely contains
examples and usage of using your application. For example:

Cobra is a CLI library for Go that empowers applications.
This application is a tool to generate the needed files
to quickly create a Cobra application.`,
	// Uncomment the following line if your bare application
	// has an action associated with it:
	Run: func(cmd *cobra.Command, args []string) {
		for _, arg := range args {
			anonymize(arg)
		}
	},
}

var output_dir = "."
var in_place = false

func anonymize(file string) {

	ds, err := dicom.ParseFile(file, nil)
	if err != nil {
		log.Error(err)
	}

	// Set the output path
	var anon_fpath string
	if !in_place {
		_, fpath := filepath.Split(file)
		anon_fpath = filepath.Join(output_dir, "anon_"+fpath)
		log.Info("Processing file: ", fpath)
	} else {
		anon_fpath = file
	}

	//ds.Elements = removeTagByValue(tag.PatientName, ds.Elements)
	//ds.Elements = removeTagByValue(tag.PatientID, ds.Elements)
	//ds.Elements = removeTagByValue(tag.PatientSex					, ds.Elements)
	//ds.Elements = removeTagByValue(tag.PatientBirthDate				, ds.Elements)
	//ds.Elements = removeTagByValue(tag.PatientAddress				, ds.Elements)
	//ds.Elements = removeTagByValue(tag.PatientMotherBirthName		, ds.Elements)
	//ds.Elements = removeTagByValue(tag.PatientTelephoneNumber		, ds.Elements)
	//ds.Elements = removeTagByValue(tag.PatientReligiousPreference	, ds.Elements)
	//ds.Elements = removeTagByValue(tag.OtherPatientIDSequence		, ds.Elements)

	fields := []tag.Tag{
		tag.PatientID,
		tag.PatientSex,
		tag.PatientBirthDate,
		tag.PatientAddress,
		tag.PatientMotherBirthName,
		tag.PatientTelephoneNumber,
		tag.PatientReligiousPreference,
		tag.OtherPatientIDSequence,
	}

	for _, t := range fields {
		ds.Elements = removeTagByValue(t, ds.Elements)
	}

	f, err := os.Create(anon_fpath)
	if err != nil {
		log.Error(err)
	}

	opts := []dicom.WriteOption{
		dicom.SkipVRVerification(),
	}

	err = dicom.Write(f, ds, opts...)
	if err != nil {
		log.Error(err)
	}

}

func writeFile(d dicom.Dataset, f string) {
	wr, err := os.Create(f)
	if err != nil {
		log.Error(err)
		return
	}

	//dicom_wr := dicom.NewWriter(wr)
	//
	//for _, e := range(d.Elements) {
	//	log.Info(e.Tag)
	//	dicom_wr.WriteElement(e)
	//}

	opts := []dicom.WriteOption{
		dicom.DefaultMissingTransferSyntax(),
	}
	dicom.Write(wr, d, opts...)

	wr.Close()
}

func removeTagByValue(t tag.Tag, elems []*dicom.Element) []*dicom.Element {
	for i, e := range elems {
		if e.Tag.Equals(t) {
			return append(elems[:i], elems[i+1:]...)
		}
	}

	return elems
}

// Execute adds all child commands to the root command and sets flags appropriately.
// This is called by main.main(). It only needs to happen once to the rootCmd.
func Execute() {
	err := rootCmd.Execute()
	if err != nil {
		os.Exit(1)
	}
}

func init() {
	// Here you will define your flags and configuration settings.
	// Cobra supports persistent flags, which, if defined here,
	// will be global for your application.

	// rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default is $HOME/.fct-anonymize.yaml)")

	// Cobra also supports local flags, which will only run
	// when this action is called directly.
	rootCmd.Flags().BoolP("toggle", "t", false, "Help message for toggle")
	rootCmd.Flags().StringVar(&output_dir, "outputdir", output_dir, "Where to put the output (default current directory)")
	rootCmd.Flags().BoolVar(&in_place, "inplace", in_place, "Overwrite the previous file with anonymized version (default FALSE)")
}
