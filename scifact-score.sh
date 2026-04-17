rm -rf output.txt

cargo run --release --features t5-quantized,metal querycsv $HOME/src/xtr-warp/beir/scifact/questions.test.tsv output.txt &&\

python score.py output.txt $HOME/src/xtr-warp/beir/scifact/collection_map.json $HOME/src/xtr-warp/beir/scifact/qrels.test.json
