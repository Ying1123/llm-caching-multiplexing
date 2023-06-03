# python xsum.py --device $1 --model facebook/opt-125m --query-path exp/xsum/prompts-20.json
# python xsum.py --device $1 --model facebook/opt-1.3b --query-path exp/xsum/prompts-20.json
# python xsum.py --device $1 --model facebook/opt-2.7b --query-path exp/xsum/prompts-20.json
# python xsum.py --device $1 --model facebook/opt-6.7b --query-path exp/xsum/prompts-20.json

# python xsum.py --device $1 --model ~/llama-7b --query-path exp/xsum/prompts-20.json

# python oasst.py --device $1 --model flan-t5-xxl --query-path exp/oasst/prompts-$2.json
# python oasst.py --device $1 --model lmsys/fastchat-t5-3b-v1.0 --query-path exp/oasst/prompts-$2.json
# python oasst.py --device $1 --model /home/Ying/models/vicuna-13b-v1.1 --query-path exp/oasst/prompts-$2.json

python oasst.py --eval-path exp/oasst/$1/flan-t5-xxl_prompts-$2_output.json
# python oasst.py --eval-path exp/oasst/$1/fastchat-t5-3b-v1.0_prompts-$2_output.json
# python oasst.py --eval-path exp/oasst/$1/vicuna-13b-v1.1_prompts-$2_output.json
