# Grapevine_Botrytis_disease_model
This model is coded based on:
A Mechanistic Model of Botrytis cinerea on Grapevines That Includes Weather, Vine Growth Stage, and the Main Infection Pathways
Elisa González-Domínguez,Tito Caffi,Nicola Ciliberti,Vittorio Rossi 
Published: October 12, 2015
https://doi.org/10.1371/journal.pone.0140444

Abstract
A mechanistic model for Botrytis cinerea on grapevine was developed. The model, which accounts for conidia production on various inoculum sources and for multiple infection pathways, considers two infection periods. During the first period (“inflorescences clearly visible” to “berries groat-sized”), the model calculates: i) infection severity on inflorescences and young clusters caused by conidia (SEV1). During the second period (“majority of berries touching” to “berries ripe for harvest”), the model calculates: ii) infection severity of ripening berries by conidia (SEV2); and iii) severity of berry-to-berry infection caused by mycelium (SEV3). The model was validated in 21 epidemics (vineyard × year combinations) between 2009 and 2014 in Italy and France. A discriminant function analysis (DFA) was used to: i) evaluate the ability of the model to predict mild, intermediate, and severe epidemics; and ii) assess how SEV1, SEV2, and SEV3 contribute to epidemics. The model correctly classified the severity of 17 of 21 epidemics. Results from DFA were also used to calculate the daily probabilities that an ongoing epidemic would be mild, intermediate, or severe. SEV1 was the most influential variable in discriminating between mild and intermediate epidemics, whereas SEV2 and SEV3 were relevant for discriminating between intermediate and severe epidemics. The model represents an improvement of previous B. cinerea models in viticulture and could be useful for making decisions about Botrytis bunch rot control.

## pipelines
1. the R docker files:
   build and pull the docker image, enter into R local server, in this way, you can specify your own R version and packages
   bash docker_up.sh, localhost, then in the R console activate renv::activate(), renv::hydrate()
2. The MLflow pipeline
   -conda env create -f environment.yaml
   -conda activate myenv
   -pip install -r requirements.txt
   -install all conflicts ones manually

   

   
