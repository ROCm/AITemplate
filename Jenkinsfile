def rocmnode(name) {
    return 'rocmtest && miopen && ' + name
}

def show_node_info() {
    sh """
        echo "NODE_NAME = \$NODE_NAME"
        lsb_release -sd
        uname -r
        ls /opt/ -la
    """
}

def runShell(String command){
    def responseCode = sh returnStatus: true, script: "${command} > tmp.txt"
    def output = readFile(file: "tmp.txt")
    echo "tmp.txt contents: $output"
    return (output != "")
}

def getDockerImageName(){
    def img
    img = "${env.CK_DOCKERHUB}:ait_rocm${params.ROCMVERSION}"
    return img
}

def getDockerImage(Map conf=[:]){
    env.DOCKER_BUILDKIT=1
    def prefixpath = conf.get("prefixpath", "/opt/rocm") // prefix:/opt/rocm
    def no_cache = conf.get("no_cache", false)
    def dockerArgs = "--build-arg BUILDKIT_INLINE_CACHE=1 --build-arg PREFIX=${prefixpath} --build-arg ROCMVERSION='${params.ROCMVERSION}' "
    echo "Docker Args: ${dockerArgs}"
    def image = getDockerImageName()
    //Check if image exists 
    def retimage
    try 
    {
        echo "Pulling image: ${image}"
        retimage = docker.image("${image}")
        retimage.pull()
    }
    catch(Exception ex)
    {
        error "Unable to locate image: ${image}"
    }
    return [retimage, image]
}

def build_ait(Map conf=[:]){

    def build_cmd = """
        export ROCM_PATH=/opt/rocm
        export ROC_USE_FGS_KERNARG=0
        pip3 uninstall -y aitemplate
        cd python
        rm -rf dist build
        python3 setup.py bdist_wheel
        pip3 install dist/*.whl
        pip3 install timm
        pip3 uninstall -y torch 
        pip3 install torch --extra-index-url https://download.pytorch.org/whl/rocm5.1.1
        python3 -m pip install transformers click
        python3 -m pip install diffusers==0.11.1 accelerate
        python3 -c "import torch; print(torch.__version__)"
        """

    def cmd = conf.get("cmd", """
        ${build_cmd}
        """)

    cmd += """
        ${execute_cmd}
    """
    echo cmd
    sh cmd
}

def Run_Step(Map conf=[:]){
        show_node_info()

        env.HSA_ENABLE_SDMA=0
        checkout scm

        def image = getDockerImageName() 
        def prefixpath = conf.get("prefixpath", "/opt/rocm")

        // Jenkins is complaining about the render group 
        def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
        if (conf.get("enforce_xnack_on", false)) {
            dockerOpts = dockerOpts + " --env HSA_XNACK=1 "
        }
        def dockerArgs = "--build-arg PREFIX=${prefixpath} --build-arg ROCMVERSION='${params.ROCMVERSION}' "
        def variant = env.STAGE_NAME
        def retimage

        gitStatusWrapper(credentialsId: "${status_wrapper_creds}", gitHubContext: "Jenkins - ${variant}", account: 'ROCmSoftwarePlatform', repo: 'AITemplate') {
            try {
                (retimage, image) = getDockerImage(conf)
                withDockerContainer(image: image, args: dockerOpts) {
                    timeout(time: 5, unit: 'MINUTES'){
                        sh 'PATH="/opt/rocm/opencl/bin:/opt/rocm/opencl/bin/x86_64:$PATH" clinfo | tee clinfo.log'
                        if ( runShell('grep -n "Number of devices:.*. 0" clinfo.log') ){
                            throw new Exception ("GPU not found")
                        }
                        else{
                            echo "GPU is OK"
                        }
                    }
                }
            }
            catch (org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e){
                echo "The job was cancelled or aborted"
                throw e
            }

            withDockerContainer(image: image, args: dockerOpts + ' -v=/var/jenkins/:/var/jenkins') {
                timeout(time: 24, unit: 'HOURS')
                {
                    build_ait(conf)
					dir("examples"){
                        sh "./run_tests.sh $HF_TOKEN"
                        archiveArtifacts "resnet50.log"
                        archiveArtifacts "bert.log"
                        archiveArtifacts "vit.log"
                        archiveArtifacts "sdiff.log"
                        // stash perf files to master
                        stash name: "resnet50.log"
                        stash name: "bert.log"
                        stash name: "vit.log"
                        stash name: "sdiff.log"
                        //we will process the results on the master node
					}
                }
            }
        }
        return retimage
}

def Run_Step_and_Reboot(Map conf=[:]){
    try{
        Run_Step(conf)
    }
    catch(e){
        echo "throwing error exception while building CK"
        echo 'Exception occurred: ' + e.toString()
        throw e
    }
    finally{
        if (!conf.get("no_reboot", false)) {
            reboot()
        }
    }
}

def process_results(Map conf=[:]){
    env.HSA_ENABLE_SDMA=0
    checkout scm
    def image = getDockerImageName() 
    def prefixpath = "/opt/rocm"

    // Jenkins is complaining about the render group 
    def dockerOpts="--cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
    if (conf.get("enforce_xnack_on", false)) {
        dockerOpts = dockerOpts + " --env HSA_XNACK=1 "
    }

    def variant = env.STAGE_NAME
    def retimage

    gitStatusWrapper(credentialsId: "${status_wrapper_creds}", gitHubContext: "Jenkins - ${variant}", account: 'ROCmSoftwarePlatform', repo: 'AITemplate') {
        try {
            (retimage, image) = getDockerImage(conf)
        }
        catch (org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e){
            echo "The job was cancelled or aborted"
            throw e
        }
    }

    withDockerContainer(image: image, args: dockerOpts + ' -v=/var/jenkins/:/var/jenkins') {
        timeout(time: 1, unit: 'HOURS'){
            try{
                dir("examples"){
                    // unstash perf files to master
                    unstash "resnet50.log"
                    unstash "bert.log"
                    unstash "vit.log"
                    unstash "sdiff.log"
                    sh "python3 process_results.py"
                }
            }
            catch(e){
                echo "throwing error exception while processing performance test results"
                echo 'Exception occurred: ' + e.toString()
                throw e
            }
        }
    }
}

pipeline {
    agent none
    triggers {
        parameterizedCron(CRON_SETTINGS)
    }
    options {
        parallelsAlwaysFailFast()
    }
    parameters {
        string(
            name: 'ROCMVERSION', 
            defaultValue: '5.4.3', 
            description: 'Specify which ROCM version to use: 5.4.3 (default).')
    }
    environment{
        dbuser = "${dbuser}"
        dbpassword = "${dbpassword}"
        dbsship = "${dbsship}"
        dbsshport = "${dbsshport}"
        dbsshuser = "${dbsshuser}"
        dbsshpassword = "${dbsshpassword}"
        status_wrapper_creds = "${status_wrapper_creds}"
        HF_TOKEN = "${HF_TOKEN}"
        DOCKER_BUILDKIT = "1"
    }
    stages{
		stage("Build AITemplate")
        {
            parallel
            {
                stage("Build AIT and Run Tests")
                {
                    agent{ label rocmnode("gfx908 || gfx90a") }
                    steps{
                        Run_Step_and_Reboot(no_reboot:true, , prefixpath: '/usr/local')
                    }
                }
            }
        }
        stage("Process Performance Test Results")
        {
            parallel
            {
                stage("Process results"){
                    agent { label 'mici' }
                    steps{
                        process_results()
                    }
                }
            }
        }
    }
}