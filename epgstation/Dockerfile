FROM l3tnun/epgstation:master-debian as upstream

RUN apt update -y && apt install -y ca-certificates git g++ cmake make libboost-dev libboost-program-options-dev libboost-filesystem-dev --no-install-recommends --no-install-suggests
RUN git clone https://github.com/harusamee/tsdivider.git
RUN cmake -B tsdivider/build tsdivider
RUN cmake --build tsdivider/build
RUN cmake --install tsdivider/build

FROM archlinux:latest

RUN echo 'Server = http://ftp.tsukuba.wide.ad.jp/Linux/archlinux/$repo/os/$arch' > /etc/pacman.d/mirrorlist
RUN pacman -Syuv --noconfirm
RUN pacman -S --noconfirm nodejs-lts-hydrogen npm ffmpeg linux-firmware intel-media-driver onevpl-intel-gpu python-pip
RUN pip install --break-system-packages numpy opencv-python-headless ultralytics onnx openvino-dev

COPY --from=upstream /app /app/
COPY --from=upstream /app/client /app/client/
COPY --from=upstream /usr/local/bin/tsdivider /usr/local/bin/tsdivider

ENV LIBVA_DRIVER_NAME=iHD

EXPOSE 8888
WORKDIR /app
ENTRYPOINT ["npm"]
CMD ["start"]
