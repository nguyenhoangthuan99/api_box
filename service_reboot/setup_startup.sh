#!/bin/bash

sudo cp ./service_techpro.service /etc/systemd/system
sudo chmod +x /etc/systemd/system/service_techpro.service
chmod +x ./setup.sh
sudo systemctl daemon-reload
sudo systemctl enable service_techpro
sudo systemctl start service_techpro
