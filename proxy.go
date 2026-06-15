package main

import (
	"bytes"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"io"
	"math/big"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"time"
)

func inferenceProxyHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		settings := getSettings()

		// Auth
		if settings.InferenceAPIKey != "" {
			authHeader := r.Header.Get("Authorization")
			expected := "Bearer " + settings.InferenceAPIKey
			if authHeader != expected {
				http.Error(w, "Unauthorized", http.StatusUnauthorized)
				return
			}
		} else {
			// If no key is set, maybe deny? The prompt implies it's required if set, or just requires it.
			// "Protect the proxied '/v1/' endpoints using a dedicated 'Authorization: Bearer <API_KEY>' check."
			// We can enforce it if the settings have it.
		}

		// Path allowlist
		if len(r.URL.Path) < 4 || r.URL.Path[:4] != "/v1/" {
			http.Error(w, "Forbidden", http.StatusForbidden)
			return
		}

		if r.URL.Path == "/v1/models" && r.Method == http.MethodGet {
			handleV1Models(w, settings.RoutingTable)
			return
		}

		port := 0
		if r.Method == http.MethodPost {
			bodyBytes, err := io.ReadAll(r.Body)
			if err != nil {
				http.Error(w, "Failed to read body", http.StatusBadRequest)
				return
			}

			var payload struct {
				Model string `json:"model"`
			}
			if err := json.Unmarshal(bodyBytes, &payload); err == nil && payload.Model != "" {
				port = settings.RoutingTable[payload.Model]
			}
			r.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
		}

		if port == 0 {
			http.Error(w, "Model not found in routing table or port not configured", http.StatusNotFound)
			return
		}

		if settings.VLLMAPIKey != "" {
			r.Header.Set("Authorization", "Bearer "+settings.VLLMAPIKey)
		} else {
			r.Header.Del("Authorization")
		}

		targetURL, _ := url.Parse(fmt.Sprintf("http://127.0.0.1:%d", port))
		proxy := httputil.NewSingleHostReverseProxy(targetURL)
		proxy.FlushInterval = -1
		proxy.ServeHTTP(w, r)
	}
}

func handleV1Models(w http.ResponseWriter, table map[string]int) {
	type ModelData struct {
		Id      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	}
	var models []ModelData
	for model := range table {
		models = append(models, ModelData{
			Id:      model,
			Object:  "model",
			Created: time.Now().Unix(),
			OwnedBy: "vllm-manager",
		})
	}
	if models == nil {
		models = []ModelData{}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"object": "list",
		"data":   models,
	})
}

func ensureCerts(certFile, keyFile string) error {
	if _, err := os.Stat(certFile); err == nil {
		if _, err := os.Stat(keyFile); err == nil {
			return nil
		}
	}

	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return err
	}

	notBefore := time.Now()
	notAfter := notBefore.Add(365 * 24 * time.Hour)

	serialNumber, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 128))
	if err != nil {
		return err
	}

	host := "localhost"
	dnsNames := []string{host}
	ipAddresses := []net.IP{net.ParseIP("127.0.0.1"), net.ParseIP("::1")}

	template := x509.Certificate{
		SerialNumber: serialNumber,
		Subject: pkix.Name{
			Organization: []string{"vLLM GPU Manager"},
			CommonName:   host,
		},
		NotBefore:             notBefore,
		NotAfter:              notAfter,
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		DNSNames:              dnsNames,
		IPAddresses:           ipAddresses,
	}

	derBytes, err := x509.CreateCertificate(rand.Reader, &template, &template, &priv.PublicKey, priv)
	if err != nil {
		return err
	}

	certOut, err := os.Create(certFile)
	if err != nil {
		return err
	}
	pem.Encode(certOut, &pem.Block{Type: "CERTIFICATE", Bytes: derBytes})
	certOut.Close()

	keyOut, err := os.Create(keyFile)
	if err != nil {
		return err
	}
	pem.Encode(keyOut, &pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(priv)})
	keyOut.Close()

	return nil
}
