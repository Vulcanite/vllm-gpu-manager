package main

import "strings"

// version is set by release builds with -ldflags "-X main.version=<tag>".
var version = "dev"

func displayVersion() string {
	value := strings.TrimSpace(version)
	if value == "" {
		return "dev"
	}
	return value
}
