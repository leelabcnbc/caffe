#! /bin/bash -x


create_tvt ()
{
    submit GLOG_logtostderr=1 ../../build/tools/compute_image_mean.bin data/${1}_train/leveldb data/${1}_train/mean.binaryproto
    submit GLOG_logtostderr=1 ../../build/tools/compute_image_mean.bin data/${1}_valid/leveldb data/${1}_valid/mean.binaryproto
    submit GLOG_logtostderr=1 ../../build/tools/compute_image_mean.bin data/${1}_test/leveldb data/${1}_test/mean.binaryproto
}

submit ()
{
    #echo "got: $@"
    #jobname="ldb_`date +%y%m%d_%H%M%S`"
    jobname="mn_`date +%H%M%S`"
    echo "#! /bin/bash" >> $jobname.sh
    echo "$@" >> $jobname.sh
    chmod +x $jobname.sh
    qsub -N "$jobname" -A EvolvingAI -l nodes=1:ppn=2 -l walltime=8:00:00 -d `pwd` $jobname.sh
    sleep 1.5
}

create_tvt half0A
create_tvt half0B
create_tvt half1A
create_tvt half1B
create_tvt half2A
create_tvt half2B
create_tvt half3A
create_tvt half3B
#create_tvt half4A
#create_tvt half4B
#create_tvt half5A
#create_tvt half5B
#create_tvt half6A
#create_tvt half6B
#create_tvt half7A
#create_tvt half7B
#create_tvt half8A
#create_tvt half8B
#create_tvt half9A
#create_tvt half9B
