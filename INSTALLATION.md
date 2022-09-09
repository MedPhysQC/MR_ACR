# MR_ACR installation instruction

+ Download factory tools:  
git clone https://bitbucket.org/MedPhysNL/wadqc.git <<TOOLSDIR>>

+ Clone repository to local machine:

  git clone http://gitlab.nki.nl/m.barsingerhorn/mr_acr.git <<REPODIR>>

+ Make factory module zip file:

cd <<REPODIR>>
pipenv sync 
pipenv shell 
cd .. 
mkdir <<FACTORY_OUTPUT_DIR>>
cd <<TOOLSDIRS>>/Tools

python .\make_factory_module.py -m zip -i  <<REPODIR>>\manifest.json -d <<FACTORY_OUTPUT_DIR>>

+ SSH to WAD server and install Docker daemon:

sudo apt install docker 
usermod -a -G docker <<user>>

+ Reboot server 

+ WINSCP to WAD server. Copy contents of <<REPODIR>> directory to wad server. 

+ SSH to WAD server, cd to directory containing <<REPODIR>>. Build docker container:

docker build –t mr_acr .

 
+	In the WAD management interface:
-	Select: “Import modules/selectors”,  select “browse” and select created zip file from <<FACTORY_OUTPUT_DIR>>. Then press “Import” 

-	Select: “Module configs”, scroll down if required and press: “new“
* Add a name and description, these can be anything. 
* For module select: MR_ACR
* For datatype select: dcm_study
* Browse to config file name:  <<REPODIR>>\config\module_config.json
* Browse to meta config file name: <<REPODIR>>\config\meta_config.json
* Finally press: Submit 

-	Select: “Selectors”, scroll down if required and press: “new“
* Add name and description, these can be anything.
* Check “isactive” checkbox
* In “config” listbox, select config with name you created earlier
* Press “Submit” 

-	In the overview with selectors, go to the just created selector and press “edit” 
* In “extra rules to add” enter ‘1’ and press enter 
* In the new rule enter “Modality” for tag and “MR” for value. (Don’t enter the parenthesis)
* Press: “Submit”

+	Send a dataset to the WAD server and test functionality
