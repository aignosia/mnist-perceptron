CC = gcc
FILE_PREFIX = tensor
TARGET = $(FILE_PREFIX).so
SOURCE = $(FILE_PREFIX).c

.PHONY: clean build

build: $(FILES)
	@$(CC) -fPIC -shared -o $(TARGET) $(SOURCE)


clean:
	@rm -rf *.so
