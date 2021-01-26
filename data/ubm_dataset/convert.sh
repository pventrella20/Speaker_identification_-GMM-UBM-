#!/bin/bash

for i in *.ogg;
do
	# shellcheck disable=SC2006
	# shellcheck disable=SC2046
	ffmpeg -i "$i" -acodec pcm_s16le -ac 1 -ar 8000 `basename "$i" .ogg`.wav
done
for i in *.mp3;
do
	# shellcheck disable=SC2006
	# shellcheck disable=SC2046
	ffmpeg -i "$i" -acodec pcm_s16le -ac 1 -ar 8000 `basename "$i" .mp3`.wav
done