package api

import (
	"fmt"
	"net"
	"net/netip"
	"strings"
)

var interfaceAddrsForTest func() ([]net.Addr, error)

func interfaceAddrs() ([]net.Addr, error) {
	if interfaceAddrsForTest != nil {
		return interfaceAddrsForTest()
	}
	return net.InterfaceAddrs()
}

func ValidateLocalBindIP(value string) (netip.Addr, error) {
	value = strings.TrimSpace(value)
	if value == "" {
		return netip.Addr{}, validationError("bind IP is required")
	}
	addr, err := netip.ParseAddr(value)
	if err != nil {
		return netip.Addr{}, validationError("bind IP must be a valid IPv4 or IPv6 address")
	}
	if addr.Is4In6() {
		addr = addr.Unmap()
	}
	if addr.IsUnspecified() || addr.IsLoopback() {
		return addr, nil
	}

	addrs, err := interfaceAddrs()
	if err != nil {
		return netip.Addr{}, fmt.Errorf("failed to list local network interfaces: %v", err)
	}
	for _, local := range addrs {
		prefix, ok := addrPrefixFromNetAddr(local)
		if !ok {
			continue
		}
		localAddr := prefix.Addr()
		if localAddr.Is4In6() {
			localAddr = localAddr.Unmap()
		}
		if localAddr == addr {
			return addr, nil
		}
	}
	return netip.Addr{}, validationError("bind IP %s is not assigned to this server", addr.String())
}

func addrPrefixFromNetAddr(value net.Addr) (netip.Prefix, bool) {
	switch typed := value.(type) {
	case *net.IPNet:
		prefix, ok := netip.AddrFromSlice(typed.IP)
		if !ok {
			return netip.Prefix{}, false
		}
		ones, bits := typed.Mask.Size()
		if bits == 0 {
			bits = prefix.BitLen()
			ones = bits
		}
		return netip.PrefixFrom(prefix, ones), true
	case *net.IPAddr:
		addr, ok := netip.AddrFromSlice(typed.IP)
		if !ok {
			return netip.Prefix{}, false
		}
		return netip.PrefixFrom(addr, addr.BitLen()), true
	default:
		prefix, err := netip.ParsePrefix(value.String())
		if err == nil {
			return prefix, true
		}
		addr, err := netip.ParseAddr(value.String())
		if err != nil {
			return netip.Prefix{}, false
		}
		return netip.PrefixFrom(addr, addr.BitLen()), true
	}
}

func HostPort(host string, port int) string {
	return net.JoinHostPort(host, fmt.Sprintf("%d", port))
}
