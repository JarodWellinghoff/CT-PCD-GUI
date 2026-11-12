package util

import (
	"encoding/json"
	log "github.com/sirupsen/logrus"
	"net/http"
	"os"
	"strings"
	"time"
	"path/filepath"
)

var TELEM_URL_1 string //= "http://localhost:8080/api/wljkmhnefczc/auth"
var TELEM_URL_2 string //= "http://localhost:8080/api/addrecord"
//var TELEM_USER string //= "a"
//var TELEM_PASS string //= "b"

func FireTelemetry(no_telemetry bool, c chan struct{}) {
	defer func() { c <- struct{}{} }()

	if no_telemetry || TELEM_URL_1 == "" || TELEM_URL_2 == "" {
		return
	}

	// Request and handle current credentials
	resp, err := http.Get(TELEM_URL_1)
	if err != nil {
		log.Debug("credential fetch failed: ", err)
		return
	}

	cp := &CredPacket{}
	err = json.NewDecoder(resp.Body).Decode(cp)
	if err != nil {
		log.Debug("credential decode failed: ", err)
		return
	}

	u, p, err := cp.Decode()
	if err != nil {
		log.Debug("base64 decode failed: ", err)
	}

	// Use credential to send telemetry packet
	var cmd string
	if len(os.Args) > 1 {
		cmd = os.Args[1]
	}

	telem_packet := struct {
		Time    string `json:"time"`
		Exe     string `json:"exe"`
		Cmd     string `json:"cmd"`
		Version string `json:"version"`
	}{
		Time:    time.Now().UTC().Format(time.RFC1123),
		Exe:     filepath.Base(os.Args[0]),
		Cmd:     cmd,
		Version: Version(),
	}

	data, err := json.Marshal(telem_packet)
	if err != nil {
		log.Debug(err)
		return
	}

	r := strings.NewReader(string(data))
	req, err := http.NewRequest(http.MethodPost, TELEM_URL_2, r)
	if err != nil {
		log.Debug("Telemetry error: ", err)
		return
	}
	req.SetBasicAuth(u, p)

	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		log.Debug("Telemetry error: ", err)
		return
	}

	if resp.StatusCode != http.StatusOK {
		log.Debug("Telemetry error: HTTP response ", resp.StatusCode)
		return
	}

}
