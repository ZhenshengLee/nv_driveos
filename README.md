# driveos学习整理版

## 简介

如果最终安装文件是`*.gz`的，解压目录在`/drive`目录下

如果最终安装目录是`/opt`，按以下规则处理：

`/opt/nvidia/driveos/6.0.5.0/31732581/drive-*` 改成`/driveos/6.0.5.0/drive-*`

`*`包括 `drive-foundation` `drive-foundation_src` `driveinstaller`

`/opt/nvidia/driveos/6.0.5.0/31732390/drive-*` 改成`/driveos/6.0.5.0/drive-*`

`*`包括 `drive-linux` `drive-linux_src`

`/opt/nvidia/driveos/common` 改成 `/driveos/common`


## 本库包含deb包

本库包含的是核心包，来自`nv-driveos-repo-sdk-linux-6.0.5.0-31732731_6.0.5.0_amd64.deb`

另外包含一个top层包 `nvsci_pkg_x86_64_embedded-6.0.4.0_20220701_30765060.deb`

具体包括如下包

```sh
nv-driveos-linux-nv-minimal-sdk-6.0.5.0-31732390_6.0.5.0-31732390_amd64.deb
# meta package: linux-nv-minimal-sdk
nv-driveos-linux-nv-minimal-sdk-usermode-multimedia-6.0.5.0-31732390_6.0.5.0-31732390_amd64.deb
# nvmedia, nvsipl
nv-driveos-linux-nv-minimal-sdk-usermode-nvstreams-6.0.5.0-31732390_6.0.5.0-31732390_amd64.deb
# nvstreams

# 暂时不关注的包

# 忽略包
nv-driveos-linux-nv-minimal-sdk-base-6.0.5.0-31732390_6.0.5.0-31732390_amd64.deb
nv-driveos-linux-nv-minimal-sdk-usermode-buildfs-6.0.5.0-31732390_6.0.5.0-31732390_amd64.deb
nv-driveos-linux-nv-minimal-sdk-usermode-compute-6.0.5.0-31732390_6.0.5.0-31732390_amd64.deb
nv-driveos-linux-nv-minimal-sdk-usermode-graphics-6.0.5.0-31732390_6.0.5.0-31732390_amd64.deb
```

## 其他deb包简介

### repo-drive-oss

```sh
nv-driveos-linux-oss-minimal-sdk-6.0.5.0-31732390_6.0.5.0-31732390_amd64.deb
# metapackage: oss-sdk
nv-driveos-linux-oss-minimal-sdk-core-6.0.5.0-31732390_6.0.5.0-31732390_amd64.deb
nv-driveos-linux-oss-minimal-sdk-kernel-6.0.5.0-31732390_6.0.5.0-31732390_amd64.deb
nv-driveos-linux-oss-minimal-sdk-kernel-rt-patches-6.0.5.0-31732390_6.0.5.0-31732390_amd64.deb
nv-driveos-linux-oss-src-6.0.5.0-31732390_6.0.5.0-31732390_amd64.deb
```

### repo-drive-foundation

```sh
nv-driveos-build-sdk-linux-6.0.5.0-31732731_6.0.5.0-31732731_amd64.deb
# top meta package: build-sdk

nv-driveos-common-build-fs-17.1.7-5a41bab6_17.1.7-5a41bab6_amd64.deb
# build fs
nv-driveos-common-copytarget-1.4.8-f1c1947f_1.4.8-f1c1947f_amd64.deb
# copy target py

nv-driveos-foundation-release-sdk-6.0.5.0-31732581_6.0.5.0-31732581_amd64.deb
# meta package: doundation-release-sdk
nv-driveos-foundation-release-sdk-core-flash-data-6.0.5.0-31732581_6.0.5.0-31732581_amd64.deb
nv-driveos-foundation-release-sdk-core-flash-tools-6.0.5.0-31732581_6.0.5.0-31732581_amd64.deb
nv-driveos-foundation-release-sdk-flash-data-6.0.5.0-31732581_6.0.5.0-31732581_amd64.deb
nv-driveos-foundation-release-sdk-flash-tools-6.0.5.0-31732581_6.0.5.0-31732581_amd64.deb
# flash data and tools
nv-driveos-foundation-release-sdk-ist-6.0.5.0-31732581_6.0.5.0-31732581_amd64.deb
nv-driveos-foundation-release-sdk-platform-config-6.0.5.0-31732581_6.0.5.0-31732581_amd64.deb
nv-driveos-foundation-release-sdk-virtualization-6.0.5.0-31732581_6.0.5.0-31732581_amd64.deb
# firmware

nv-driveos-foundation-toolchains-9.3.0_9.3.0_amd64.deb
# meta package: toolchains
nv-driveos-foundation-gcc-bootlin-gcc9.3-aarch64--glibc--stable-2022.03-1_9.3.0_amd64.deb
nv-driveos-foundation-gcc-bootlin-gcc9.3-armv5-eabi--glibc--stable-2020.08-1_9.3.0_amd64.deb
nv-driveos-foundation-gcc-bootlin-gcc9.3-armv7-eabihf--glibc--stable-2020.08-1_9.3.0_amd64.deb
# gcc工具

nv-driveos-linux-driveos-core-ubuntu-20.04-rfs-6.0.5.0-31732390_6.0.5.0-31732390_amd64.deb
nv-driveos-linux-initramfs-6.0.5.0-31732390_6.0.5.0-31732390_amd64.deb
# os related
```

## top层其他包简介

```sh

nv-driveos-foundation-driveinstaller-6.0.5.0-31732581_6.0.5.0-31732581_amd64.deb
# 一些安装脚本

nv-driveos-foundation-oss-src-6.0.5.0-31732581_6.0.5.0-31732581_amd64.deb
# doundation-src # gcc and buildroot # jq-json processor # e2fsprog # 3rd-dtc

nv-driveos-foundation-p3663-specific-6.0.5.0-31732581_6.0.5.0-31732581_amd64.deb
nv-driveos-foundation-p3710-specific-6.0.5.0-31732581_6.0.5.0-31732581_amd64.deb

nv-driveos-linux-driveos-oobe-desktop-ubuntu-20.04-rfs-6.0.5.0-31732390_6.0.5.0-31732390_amd64.deb
# ubuntu镜像

nv-driveos-linux-vksc-dev-6.0.5.0-31732390_6.0.5.0-31732390_amd64.deb
nv-driveos-linux-vksc-ecosystem-6.0.5.0-31732390_6.0.5.0-31732390_amd64.deb
# vksc库



```
