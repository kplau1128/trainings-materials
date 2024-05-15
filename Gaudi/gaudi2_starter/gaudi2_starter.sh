#!/bin/sh

export PHY_CPU_COUNT=$(lscpu --all --parse=CORE,SOCKET | grep -Ev "^#" | sort -u | wc -l)
export PHY_HPU_COUNT=$(ls /dev/hl? | wc -l)
export MPI_PE=$(($PHY_CPU_COUNT/$PHY_HPU_COUNT))
export RELEASE=1.14.0
export HTTPS_PROXY=http://proxy-chain.intel.com:912/
export HTTP_PROXY=http://proxy-chain.intel.com:911
export PYTHONPATH="/root/Model-References:$PYTHONPATH"
export PYTHON=python3.8
exit_status()
{
  _status=$?
  _func=$1
  if (( _status != 0 )); then
    echo "Exiting $_func with status $_status"
    exit
  fi
}

fn_exists()
{
    _func=$1
    LC_ALL=C type $_func 2>/dev/null | head -1 | grep -q 'is a function'
}

bert_config_tf()
{
  echo "Inside bert_config_tf"
  rm -rf /tmp/*
  export TF_BF16_CONVERSION=/root/Model-References/TensorFlow/nlp/bert/bf16_config/bert.json
  pip3 install -r /root/Model-References/TensorFlow/nlp/bert/requirements.txt
  mkdir -p /tmp/pretraining/phase_1
  mkdir -p /tmp/pretraining/phase_2
}

bert_config_pt()
{
  echo "Running configuration for pytorch bert model"
  # pip3 install datasets==1.12.0
  rm -rf /tmp/*
  pip3 install -r "/root/Model-References/internal/PyTorch/nlp/finetuning/huggingface/bert/transformers/examples/pytorch/question-answering/requirements.txt"
  pip3 install -r "/root/Model-References/internal/PyTorch/nlp/finetuning/huggingface/bert/transformers/examples/pytorch/text-classification/requirements.txt"
  pip3 install "/root/Model-References/internal/PyTorch/nlp/finetuning/huggingface/bert/transformers/."
  pip install -r /root/Model-References/PyTorch/nlp/bert/requirements.txt
  mkdir -p /tmp/BERT_PRETRAINING/results
}

resnet_config_tf()
{
  echo "Running configuration for resnet model"
  export PT_HPU_POOL_STRATEGY=3
  #pip install hpu-media-loader==1.4.0.435
  pip install Pillow
  pip install colorlog
  pip install columnar

  pip3 install -r /root/Model-References/TensorFlow/computer_vision/Resnets/resnet_keras/requirements.txt
}

resnet_config_pt()
{
  echo "running configuration for pytorch resnet model"
  pip3 install -r /root/Model-References/PyTorch/computer_vision/classification/torchvision/requirements.txt
}


tf_resnet50_1x()
{
  if [ ! -d "/root/Model-References" ]; then
    git clone https://github.com/HabanaAI/Model-References.git /root/Model-References
  fi
  echo "Running TF Resnet50 model on single card"
  resnet_config_tf
  python3 /root/Model-References/TensorFlow/computer_vision/Resnets/resnet_keras/resnet_ctl_imagenet_main.py \
          --optimizer=LARS --dtype=bf16 --batch_size=256 \
          --jpeg_data_dir=/data2/pytorch/imagenet/ILSVRC2012/ \
          --data_loader_image_type=bf16 --model_dir=/tmp/resnet/ --steps_per_loop=200 \
          --train_steps=8000 --enable_tensorboard --experimental_preloading 2>&1 | tee /tmp/tf_resnet50_1x.txt
  python3 get_resnet50_perf.py /tmp/tf_resnet50_1x.txt 100
  exit_status "${FUNCNAME[0]}"

}

tf_resnet50_8x()
{
  if [ ! -d "/root/Model-References" ]; then
    git clone https://github.com/HabanaAI/Model-References.git /root/Model-References
  fi
  echo "Running TF Resnet50 model on 8 card"
  resnet_config_tf
  mpirun --allow-run-as-root --bind-to core --map-by socket:PE=${MPI_PE} --np 8 \
  python3 /root/Model-References//TensorFlow/computer_vision/Resnets/resnet_keras/resnet_ctl_imagenet_main.py \
    --batch_size=256 --jpeg_data_dir=/data2/pytorch/imagenet/ILSVRC2012/ \
    --data_loader_image_type=bf16 --dtype=bf16 --epochs_between_evals=5 --log_steps=200 --model_dir=/tmp/resnet/ \
    --steps_per_loop=200 --train_epochs=2 --optimizer=LARS --base_learning_rate=9.5 --warmup_epochs=3 \
    --lr_schedule=polynomial --label_smoothing=0.1 --weight_decay=0.0001 --enable_tensorboard --experimental_preloading \
    --single_l2_loss_op --use_horovod 2>&1 | tee /tmp/tf_resnet50_8x.txt
  python3 get_resnet50_perf.py /tmp/tf_resnet50_8x.txt 100 
  exit_status "${FUNCNAME[0]}"

}

optimum_habana_GPT2_8x()
{
        if [ ! -d "/root/optimum-habana" ]; then
            git clone https://github.com/huggingface/optimum-habana.git /root/optimum-habana
        fi
        pip install optimum[habana]
        cd /root/optimum-habana/examples/language-modeling/
        pip install -r requirements.txt
	python ../gaudi_spawn.py \
    --world_size 8 --use_mpi run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm \
    --gaudi_config_name Habana/gpt2 \
    --use_habana \
    --use_lazy_mode \
    --use_hpu_graphs_for_inference \
    --gradient_checkpointing \
    --use_cache False \
    --throughput_warmup_steps 3
    exit_status "${FUNCNAME[0]}"

}

optimum_habana_GPT2_1x()
{
        if [ ! -d "/root/optimum-habana" ]; then
            git clone https://github.com/huggingface/optimum-habana.git /root/optimum-habana
        fi
        pip install optimum[habana]
        cd /root/optimum-habana/examples/language-modeling/
        pip install -r requirements.txt
	python run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm \
    --gaudi_config_name Habana/gpt2 \
    --use_habana \
    --use_lazy_mode \
    --use_hpu_graphs_for_inference \
    --throughput_warmup_steps 3
    exit_status "${FUNCNAME[0]}"

}

pt_hf_ds_roberta_large_8x()
{
        if [ ! -d "/root/optimum-habana" ]; then
            git clone https://github.com/huggingface/optimum-habana.git /root/optimum-habana
        fi
        pip install optimum[habana]
        cd /root/optimum-habana/examples/question-answering
        pip install -r requirements.txt
        pip install git+https://github.com/HabanaAI/DeepSpeed.git@$RELEASE
        python ../gaudi_spawn.py \
    --world_size 8 --use_deepspeed run_qa.py \
    --model_name_or_path roberta-large \
    --gaudi_config_name Habana/roberta-large \
    --dataset_name squad \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 8 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir /tmp/SQUAD/ \
    --use_habana \
    --use_lazy_mode \
    --use_hpu_graphs_for_inference \
    --throughput_warmup_steps 3 \
    --deepspeed /root/hf_deepspeed_gpt2.config \
    --overwrite_output_dir
    exit_status "${FUNCNAME[0]}"

}

# PT Inference models ####################################################
pt_bloom_inference_7b1 ()
{
  if [ ! -d "/root/Model-References" ]; then
    git clone https://github.com/HabanaAI/Model-References.git /root/Model-References
  fi
  cd /root/Model-References/PyTorch/nlp/bloom
  python3 -m pip install -r requirements.txt
  pip install git+https://github.com/HabanaAI/DeepSpeed.git@$RELEASE
  export PYTHONPATH=$PYTHONPATH:/root/Model-References
  export PYTHON=/usr/bin/python3.8
  mkdir -p checkpoints
  python3 utils/fetch_weights.py --weights ./checkpoints --model bigscience/bloom-7b1
  ./bloom.py --weights ./checkpoints --model bloom-7b1 --batch_size=8 --dtype bf16 --options "max_length=2048,num_beams=1" "It was the best of times" --repeat 10

}
model=$1

if fn_exists $model; then
  # clear_logs
  $model
elif [ -z $model ]; then
  echo "usage: bash gaudi2.sh [model_name]"
  echo "#################################################"
  echo "model_name option for TensorFlow docker : tf_resnet50_1x, tf_resnet50_8x "
  echo "#################################################"
  echo "Optimum habana option for PyTorch docker 8 card : optimum_habana_GPT2_8x, pt_hf_ds_roberta_large_8x "
  echo "#################################################"
  echo "Optimum habana option for PyTorch docker 1 card : optimum_habana_GPT2_1x "
  echo "#################################################"
  echo "Inference model_name option for PyTorch docker : pt_bloom_inference_7b1  "
  echo "#################################################"
else
  echo "Error: Model $model is not supported"
fi

