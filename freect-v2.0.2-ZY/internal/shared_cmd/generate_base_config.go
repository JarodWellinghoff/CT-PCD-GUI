package shared_cmd

import (
	"fmt"
	log "github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"gitlab.com/freect/freect/pkg/image"
	"gitlab.com/freect/freect/pkg/raw"
	"gitlab.com/freect/freect/pkg/recon"
	"gopkg.in/yaml.v3"
	"reflect"
	"strings"
)

func GenerateBaseConfigCommand(c *recon.ReconConfig, rootCmd *cobra.Command) {

	var cmdGenBaseConfig = &cobra.Command{
		Use:   "generate-base-config",
		Short: "Print a base config file to stdout for a specified raw data set",
		Long:  "",
		Run: func(cmd *cobra.Command, args []string) {

			c, err := recon.GenerateBaseConfig(c)
			if err != nil {
				log.Fatal("Unable to generate a base configuration file: ", err)
			}

			// Finally, print the populated config to stdout
			data, err := yaml.Marshal(c)
			if err != nil {
				log.Fatal(err)
			}

			fmt.Println(string(data))
		},
	}

	GenerateInputFlags(c.RawInput, cmdGenBaseConfig)
	GenerateOutputFlags(c.ImageOutput, cmdGenBaseConfig)

	rootCmd.AddCommand(cmdGenBaseConfig)
}

func GenerateInputFlags(in *raw.RawInputConfig, cmd *cobra.Command) {

	input_fields := reflect.VisibleFields(reflect.TypeOf(*in))

	for _, f := range input_fields {
		v := reflect.ValueOf(in).Elem().FieldByName(f.Name) // Represents the underlying string as reflect.Value
		p := v.Addr().Interface().(*string)
		flag := "input-" + strings.ToLower(f.Name)
		default_val := v.String()
		usage := "set raw input type " + f.Name
		cmd.Flags().StringVar(p, flag, default_val, usage)
	}

}

func GenerateOutputFlags(out *image.ImageOutputConfig, cmd *cobra.Command) {
	input_fields := reflect.VisibleFields(reflect.TypeOf(*out))
	for _, f := range input_fields {
		v := reflect.ValueOf(out).Elem().FieldByName(f.Name) // Represents the underlying string as reflect.Value

		if f.Name == "RescaleToHU" || f.Name == "AngleOffset" {
			continue
		}

		p := v.Addr().Interface().(*string)
		flag := "output-" + strings.ToLower(f.Name)
		default_val := v.String()
		usage := "set output image type " + f.Name
		cmd.Flags().StringVar(p, flag, default_val, usage)
	}
}
