@echo off
echo Starting overnight Bao experiments...

:: 1. Run the full robust model first
echo Training full 48-arm model...
python main.py --num-arms 48

:: BACKUP THE ROBUST MODEL BEFORE IT GETS OVERWRITTEN!
copy models\bao_imdb.pt models\ROBUST_48_bao_imdb.pt
echo Robust model backed up successfully.

:: 2. Run the baseline (1 arm = Native PostgreSQL)
echo Generating 1-arm baseline metrics...
python main.py --num-arms 1

:: 3. Run the restricted Bao action spaces for Figure 12
echo Generating restricted action space metrics...
python main.py --num-arms 5
python main.py --num-arms 15
python main.py --num-arms 25
python main.py --num-arms 35
python main.py --num-arms 45

echo All experiments complete! Your robust model is saved as ROBUST_48_bao_imdb.pt