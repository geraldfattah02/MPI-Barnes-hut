CC = mpic++
CFLAGS = -O3 -std=c++11
TARGET = nbody
LIBRARIES_BEFORE = -L/usr/local/lib -lGL -lGLU -lglut -lGLEW -lXxf86vm -lXrandr
LIBRARIES_AFTER = -ldl -lX11 -lpthread

LDLIBS = $(LIBRARIES_BEFORE) /usr/local/lib/libglfw3.a $(LIBRARIES_AFTER)

all: $(TARGET)

$(TARGET): main.cpp
	$(CC) main.cpp -o $(TARGET) $(CFLAGS) $(LDLIBS)

clean:
	rm -f $(TARGET)