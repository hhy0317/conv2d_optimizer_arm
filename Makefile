# 设置编译器
CC = gcc

# 设置编译选项
# CFLAGS = -Wall -Werror
CFLAGS = -Wall

# 设置头文件
CFLAGS += -I ./inc

CFLAGS += -O2 -mfpu=neon -mfloat-abi=hard

# 定义源文件
SRCS = ./src/main.c \
       ./src/conv2d_process.c \
       ./src/conv2d_process_fixed.c \
	   ./src/gemm.c

# 定义链接库
LIBS = -lm \

OBJ = $(SRCS:.c=.o)

# 定义目标程序名
TARGET = output/conv2d_program

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $(TARGET) $(LIBS)

# 编译目标：将 .c 文件编译为 .o 文件
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(OBJ) $(TARGET)
