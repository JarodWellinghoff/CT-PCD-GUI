package util

import (
	"encoding/base64"
	"strings"
)

type CredPacket struct {
	Cred string `json:"cred"`
}

func (c *CredPacket) Decode() (string, string, error) {
	data := make([]byte, 4096)

	n, err := base64.StdEncoding.Decode(data, []byte(c.Cred))
	if err != nil {
		return "", "", err
	}

	v := strings.Split(string(data[:n]), ":")
	return v[0], v[1], nil
}

func (c *CredPacket) Encode(u string, p string) {
	c.Cred = base64.StdEncoding.EncodeToString([]byte(u + ":" + p))
}
