all: genetic geneticViewer

CFLAGS=-O2 -lm -std=c99 -w
GFLAGS=-lGL -lglut -lGLU

genetic: genetic.c
	gcc genetic.c -o genetic $(CFLAGS)

geneticViewer: geneticViewer.c
	gcc geneticViewer.c -o geneticViewer $(CFLAGS) $(GFLAGS)
	
exec:	genetic geneticViewer
	./genetic > data && ./geneticViewer data

clean:
	rm *.o genetic geneticViewer
