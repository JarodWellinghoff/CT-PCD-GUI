package util

var FCT_VERSION_NUMBER string = ""

func Version() string {

	if FCT_VERSION_NUMBER == "" {
		return "FreeCT-custom-build-v2.0.0" // May not be up to date
	}

	return "FreeCT " + FCT_VERSION_NUMBER
}
