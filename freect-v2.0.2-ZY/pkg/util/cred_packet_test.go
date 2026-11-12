package util

import (
	"testing"
)

func TestCredPacket(t *testing.T) {
	user := "hello"
	pass := "johnson"

	c := &CredPacket{}
	c.Encode(user, pass)
	t.Log(c.Cred)

	u, p, err := c.Decode()
	if err != nil {
		t.Error(err)
	}

	if u != user {
		t.Error(u, "!=", user)
	}

	if p != pass {
		t.Error(p, "!=", pass)
	}

}
