package shared_cmd

import (
	"fmt"
	log "github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"gitlab.com/freect/freect/pkg/recon"
	"gopkg.in/yaml.v3"
)

func GenerateEmptyConfigCommand(c *recon.ReconConfig, rootCmd *cobra.Command) {
	var cmdGenEmptyConfig = &cobra.Command{
		Use:   "generate-empty-config",
		Short: "Print an empty config file to stdout",
		Long:  "",
		Run: func(cmd *cobra.Command, args []string) {
			data, err := yaml.Marshal(c)
			if err != nil {
				log.Fatal(err)
			}

			fmt.Println(string(data))
		},
	}
	rootCmd.AddCommand(cmdGenEmptyConfig)
}
