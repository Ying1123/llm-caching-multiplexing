# python oasst.py --gen-query 4

# python oasst.py --device $1 --model flan-t5-small --query-path exp/oasst_flan_t5/prompts-4.json
# python oasst.py --device $1 --model flan-t5-base --query-path exp/oasst_flan_t5/prompts-4.json
# python oasst.py --device $1 --model flan-t5-large --query-path exp/oasst_flan_t5/prompts-4.json
# python oasst.py --device $1 --model flan-t5-xl --query-path exp/oasst_flan_t5/prompts-4.json
# python oasst.py --model flan-t5-xxl --query-path exp/oasst_flan_t5/prompts-4.json

# python oasst.py --eval-path exp/oasst_flan_t5/t4/flan-t5-small_prompts-4_output.json
# python oasst.py --eval-path exp/oasst_flan_t5/a100/flan-t5-large_prompts-20_output.json
# python oasst.py --eval-path exp/oasst_flan_t5/a100/flan-t5-xl_prompts-20_output.json
# python oasst.py --eval-path exp/oasst_flan_t5/a100/flan-t5-xxl_prompts-20_output.json

python xsum.py --eval-path exp/xsum/t4/llama-7b_prompts-20_output.json
