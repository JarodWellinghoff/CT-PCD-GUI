package shared_cmd

import (
	"github.com/spf13/cobra"
	"gitlab.com/freect/freect/pkg/util"
	"fmt"
)

func AddVersionCommand(rootCmd *cobra.Command) {
	var cmdVersion = &cobra.Command{
		Use:   "version",
		Short: "Print the current version of FreeCT",
		Long:  "",
		Run: func(cmd *cobra.Command, args []string) {
			fmt.Println(util.Version())
		},
	}

	rootCmd.AddCommand(cmdVersion)
}