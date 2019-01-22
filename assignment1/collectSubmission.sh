rm -f assignment1.zip 
zip -r assignment1.zip . -x "*cs175/datasets*" "*.ipynb_checkpoints*" "*README.md" "*collectSubmission.sh" ".env/*"
