from tokenize import Name
import gpt_2_simple as gpt2
from datetime import datetime

def novel():   
    try:     
        #telecharger le fichier en local
        filer_book = "pride_and_prejudice.txt"

        #limiter les parametres de telechargement a 500M
        model = gpt2.download_gpt2(model_name='124M')

        #installer google drive
        gdrive = gpt2.mount_gdrive()

        try:
            #authoriser la connexion a google drive pour récupérer le fichier
            authorize = gpt2.copy_file_from_gdrive(filer_book)
        except NameError:
            print("Erreur de connexion a google drive")

        #crée une nouvelle session pour crée la "nouvelle"
        sess = gpt2.start_tf_sess()

        #à partir de la session definir tous les parmetre
        gpt2.finetune(sess,
                    dataset=filer_book,
                    model_name='124M',
                    steps=1000,
                    restore_from='fresh',
                    run_name='run1',
                    print_every=10,
                    sample_every=200,
                    save_every=500)


        #crée un checkpoint dans le fichier nommer run1
        gpt2.copy_checkpoint_from_gdrive(run_name='run1')

        #lui liée un code temps
        gen_file = 'gpt2_gentext_{:%Y%m%d_%H%M%S}.txt'.format(datetime.utcnow())


        gpt2.generate_to_file(
            sess,
            destination_path=gen_file,
            temperature=0.7,
            nsamples=100,
            batch_size=20)
            
    except NameError:
        print("Erreur de la nouvelle")


        #télécharger la nouvelle
        filer_book.download(gen_file)

