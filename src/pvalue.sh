set -e
for x in $(seq "$2" "$3"); do
    python3 pvalue.py "$1" "$x" | grep "LOG"
done
