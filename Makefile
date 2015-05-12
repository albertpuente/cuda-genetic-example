all: genetic geneticViewer

CFLAGS=-O2 -lm -std=c99 -w
GFLAGS=-lGL -lglut -lGLU

genetic: genetic.c
	gcc genetic.c -o genetic $(CFLAGS)

geneticViewer: geneticViewer.c
	gcc geneticViewer.c -o geneticViewer $(CFLAGS) $(GFLAGS)
	
show:	genetic geneticViewer
	./genetic > data && ./geneticViewer data

exec:	genetic geneticViewer
	./genetic exec
	
clean:
	rm *.o genetic geneticViewer
