clean:
	rm -rf ./checkpoints/*
	rm -rf ./results/*
	rm -rf ./test_results/*


run:
	make clean
	sh ./mytask.sh
