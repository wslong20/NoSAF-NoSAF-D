python main_exp.py --dataset Cora --epochs 300--dropout 0.9 --lr 0.006 --weight_decay 0.01 --input_map_numlayer 2 --output_map_numlayer 1  --backbone_numlayer 6 --gnn_in_features 128 --learner_numlayer 2 --learner_hid_features 32
python main_exp.py --dataset CiteSeer --epochs 500 --dropout 0.9 --lr 0.009 --weight_decay 0.0005 --input_map_numlayer 2 --output_map_numlayer 1  --backbone_numlayer 6 --gnn_in_features 64 --learner_hid_features 32
python main_exp.py --dataset PubMed --epochs 300 --dropout 0.8 --lr 0.007 --weight_decay 0.002 --input_map_numlayer 2 --output_map_numlayer 1  --backbone_numlayer 8 --gnn_in_features 64 --learner_numlayer 2 --learner_hid_features 4
python main_exp.py --dataset CoraFull --epochs 500 --dropout 0.5 --lr 0.004 --weight_decay 0.0 --input_map_numlayer 2 --output_map_numlayer 1 --backbone_numlayer 8 --gnn_in_features 256 --learner_numlayer 2 --learner_hid_features 16
python main_exp.py --dataset Computers --epochs 500 --dropout 0.8 --lr 0.008 --weight_decay 0.0 --input_map_numlayer 2 --output_map_numlayer 1 --backbone_numlayer 8 --gnn_in_features 64 --learner_numlayer 2 --learner_hid_features 4
python main_exp.py --dataset Photo --epochs 500 --dropout 0.8 --lr 0.0005 --weight_decay 0.001 --input_map_numlayer 2 --output_map_numlayer 1 --backbone_numlayer 8 --gnn_in_features 128 --learner_numlayer 2 --learner_hid_features 4
python main_exp.py --dataset CS --epochs 500 --dropout 0.9 --lr 0.005 --weight_decay 0.0 --input_map_numlayer 2 --output_map_numlayer 1  --backbone_numlayer 6 --gnn_in_features 256 --learner_numlayer 2 --learner_hid_features 8
python main_exp.py --dataset Physics --epochs 500 --dropout 0.9 --lr 0.008 --weight_decay 0.0005 --input_map_numlayer 2 --output_map_numlayer 1  --backbone_numlayer 12 --gnn_in_features 128 --learner_numlayer 2 --learner_hid_features 32
python main_exp.py --dataset WikiCS --epochs 500 --dropout 0.7 --lr 0.02 --weight_decay 0.0 --input_map_numlayer 2 --output_map_numlayer 1  --backbone_numlayer 8 --gnn_in_features 128 --learner_numlayer 2 --learner_hid_features 16

python main_exp.py --dataset Cornell --epochs 500 --dropout 0.8 --lr 0.01 --weight_decay 0.001 --input_map_numlayer 2 --output_map_numlayer 2  --backbone_numlayer 16 --gnn_in_features 64 --learner_numlayer 2 --learner_hid_features 16
python main_exp.py --dataset Texas --epochs 300 --dropout 0.8 --lr 0.01 --weight_decay 0.008 --input_map_numlayer 1 --output_map_numlayer 2  --backbone_numlayer 8 --gnn_in_features 64 --learner_numlayer 2 --learner_hid_features 4
python main_exp.py --dataset Wisconsin --epochs 300 --dropout 0.8 --lr 0.008 --weight_decay 0.004 --input_map_numlayer 2 --output_map_numlayer 2  --backbone_numlayer 10 --gnn_in_features 64 --learner_numlayer 2 --learner_hid_features 4
python main_exp.py --dataset Chameleon --epochs 300 --dropout 0.0 --lr 0.003 --weight_decay 0.0 --input_map_numlayer 1 --output_map_numlayer 2  --backbone_numlayer 16 --gnn_in_features 128 --learner_numlayer 2 --learner_hid_features 16
python main_exp.py --dataset Squirrel --epochs 500 --dropout 0.2 --lr 0.002 --weight_decay 0.0 --input_map_numlayer 1 --output_map_numlayer 2  --backbone_numlayer 10 --gnn_in_features 128 --learner_numlayer 2 --learner_hid_features 32
python main_exp.py --dataset Actor --epochs 500 --dropout 0.9 --lr 0.001 --weihgt_decay 0.0 --input_map_numlayer 2 --output_map_numlayer 2 --backbone_numlayer 10 --gnn_in_features 128 --learner_numlayer 2 --learner_hid_features 32



