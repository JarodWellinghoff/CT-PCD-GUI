/*
Copyright © 2022 NAME HERE <EMAIL ADDRESS>

*/
package cmd

import (
	"fmt"

	log "github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"gitlab.com/freect/freect/pkg/raw"
	"reflect"
	"strings"
)

// convertCmd represents the convert command
var convertCmd = &cobra.Command{
	Use:   "convert",
	Short: "Convert raw data between formats",
	Long: `Convert raw data between supported file formats
    - Only one input and one output format can be specified.
    - To convert to multiple output types, you must make multiple calls to the executable.
TODO: include a usage example.`,
	Run: func(cmd *cobra.Command, args []string) {

		// Verify that the user set exactly one input and exactly one output

	},
}

var in = &raw.RawInputConfig{}
var out = &raw.RawOutputConfig{}

func init() {
	rootCmd.AddCommand(convertCmd)

	GenerateInputFlags(in)
	GenerateOutputFlags(out)

}

func GenerateInputFlags(r *raw.RawInputConfig) {
	v := reflect.ValueOf(r).Elem()
	t := reflect.TypeOf(*r)

	for _, curr_field := range reflect.VisibleFields(t) {
		flag_string := "input." + strings.ToLower(curr_field.Name)

		field_val := v.FieldByName(curr_field.Name)

		if field_val.Kind() != reflect.String {
			log.Fatalf("Could not parse field %s as input flag (only strings accepted, not %s)", curr_field.Name, field_val.Kind().String())
		}

		p := field_val.Addr().Interface().(*string)
		convertCmd.Flags().StringVar(p, flag_string, "", fmt.Sprintf("Use input type %s", curr_field.Name))
	}
}

func GenerateOutputFlags(r *raw.RawOutputConfig) {
	v := reflect.ValueOf(r).Elem()
	t := reflect.TypeOf(*r)

	for _, curr_field := range reflect.VisibleFields(t) {
		flag_string := "output." + strings.ToLower(curr_field.Name)

		field_val := v.FieldByName(curr_field.Name)

		if field_val.Kind() != reflect.String {
			log.Fatalf("Could not parse field %s as output flag (only strings accepted, not %s)", curr_field.Name, field_val.Kind().String())
		}

		p := field_val.Addr().Interface().(*string)
		convertCmd.Flags().StringVar(p, flag_string, "", fmt.Sprintf("Save to output type %s", curr_field.Name))
	}
}
