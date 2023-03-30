# Lego Robot - Depth Estimation w/ Image Congealing
## Setup
### Requirements:
+ python3
+ pip
+ git
+ opencv
+ buildhat

### Update system:
```
sudo apt-get update && sudo apt-get upgrade
```

### Install VSCode (optional):
```
sudo apt install ./<file>.deb
sudo apt update
sudo apt install code
```

### Setup git credentials
```
$ git config --global user.name 'your user name'
$ git config --global user.password 'your password'
```

### Clone repository

### Enable PiCamera
```
$ sudo apt install raspi-config
$ sudo raspi-config
```

Select 'Interface Options' -> 'Legacy Camera Option' -> 'Enable'

### Enable Serial Ports
```
sudo raspi-config
```

> Select `Interface Options` -> `Serial Port` -> `No` -> `Yes` -> `Ok`
