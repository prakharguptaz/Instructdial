# run all sh in datasets
cd datasets
for f in *.sh; do
    echo "running $f"
    bash "$f"
done
cd ..
