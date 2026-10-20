#!/bin/sh
set -eu
umask 077
# Delegate the guest's cgroup hierarchy before starting any nested runtime.
mkdir -p /sys/fs/cgroup/init
for pid in $(cat /sys/fs/cgroup/cgroup.procs); do
  echo "$pid" > /sys/fs/cgroup/init/cgroup.procs 2>/dev/null || true
done
sed -e 's/ / +/g' -e 's/^/+/' /sys/fs/cgroup/cgroup.controllers > /sys/fs/cgroup/cgroup.subtree_control
apk add --no-cache curl e2fsprogs util-linux iproute2 iptables bash >/tmp/packages.log 2>&1
# overlayfs needs a compatible backing filesystem; Kata's virtiofs is not one.
mkdir -p /var/lib/rancher/k3s /dev/net /run/tailscale /demo
truncate -s "${K3S_DISK_GIB:-12}G" /k3s-data.ext4
mkfs.ext4 -q -F /k3s-data.ext4
test -e /dev/loop-control || mknod /dev/loop-control c 10 237
for i in $(seq 0 7); do test -e /dev/loop$i || mknod /dev/loop$i b 7 "$i"; done
mount -t ext4 -o loop /k3s-data.ext4 /var/lib/rancher/k3s
test -e /dev/kmsg || mknod /dev/kmsg c 1 11
test -e /dev/net/tun || mknod /dev/net/tun c 10 200
mount --make-rshared /
sysctl -w net.ipv4.ip_forward=1
mkdir -p /var/lib/rancher/k3s/cluster /var/lib/rancher/k3s/kubelet /demo/node
dockerd --host=unix:///var/run/docker.sock --data-root=/var/lib/rancher/k3s/docker --storage-driver=overlay2 >/demo/dockerd.log 2>&1 &
for i in $(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 1; done
docker info >/dev/null
docker pull "${K3S_IMAGE:-rancher/k3s:v1.37.1-k3s1}" >/demo/image-pull.log 2>&1
curl -fsSL --retry 3 https://pkgs.tailscale.com/stable/tailscale_1.104.1_amd64.tgz | tar xz -C /tmp
cp /tmp/tailscale_1.104.1_amd64/tailscale* /usr/local/bin/
curl -fsSL --retry 3 https://github.com/tailscale/tailcat/releases/download/v0.7.0/tailcat_0.7.0_linux_amd64.tar.gz | tar xz -C /tmp
cp /tmp/tailcat /usr/local/bin/
# A restored guest must load its saved VPN identity before tailscaled starts.
if [ -f /checkpoint/manifest.json ]; then
  cp /checkpoint/tailscaled.state /demo/tailscaled.state
fi
tailscaled --state=/demo/tailscaled.state --socket=/var/run/tailscale/tailscaled.sock > /demo/tailscaled.log 2>&1 &
cat > /usr/local/bin/kubectl <<'KUBECTL'
#!/bin/sh
exec docker exec k3s kubectl "$@"
KUBECTL
chmod +x /usr/local/bin/kubectl
touch /demo/ready
# The platform enforces max_lifetime_seconds, including longer dev sessions.
sleep infinity
