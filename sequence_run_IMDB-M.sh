time_str=`date +%Y%m%d%H%M%S`
mkdir nohup_logs/${time_str}
# 循环遍历 1-3 和 1-5
for i in {1..3};
do 
    for j in {1..5};
    do
    echo $i, 0.$j;
    python main.py --dataname IMDB-MULTI --depth 3 --rate 0.7 >> nohup_logs/${time_str}/IMDB-MULTI.log;
    done
done