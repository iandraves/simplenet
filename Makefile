# the compiler
CC = gcc

# compiler flags:
CFLAGS = -std=c11 -lm -Wall

# the build target executable:
TARGET = main

all: $(TARGET)

$(TARGET): $(TARGET).c
	$(CC) $(CFLAGS) -o $(TARGET).o $(TARGET).c

clean:
	$(RM) $(TARGET)