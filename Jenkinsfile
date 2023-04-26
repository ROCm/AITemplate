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
        python3 -c "import torch; print(torch.__version__)"
        """

    def cmd = conf.get("cmd", """
        ${build_cmd}
        """)

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
                        if (params.RUN_FULL_QA){
                            sh "./run_qa.sh $HF_TOKEN ${env.BRANCH_NAME} ${NODE_NAME} ${params.ROCMVERSION}"
                        }
                        else{
                            sh "./run_tests.sh $HF_TOKEN ${env.BRANCH_NAME} ${NODE_NAME} ${params.ROCMVERSION}"
                        }
                    }
                    dir("examples/01_resnet-50"){
                        archiveArtifacts "01_resnet50.log"
                        stash includes: "01_resnet50.log", name: "01_resnet50.log"
                    }
                    dir("examples/03_bert"){
                        archiveArtifacts "03_bert.log"
                        stash includes: "03_bert.log", name: "03_bert.log"
                    }
                    dir("examples/04_vit"){
                        archiveArtifacts "04_vit.log"
                        stash includes: "04_vit.log", name: "04_vit.log"
                    }
                    dir("examples/05_stable_diffusion/"){
                        archiveArtifacts "05_sdiff.log"
                        stash includes: "05_sdiff.log", name: "05_sdiff.log"
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
                    // clean up any old logs, then unstash perf files to master
                    sh "rm -rf *.log"
                    unstash "01_resnet50.log"
                    unstash "03_bert.log"
                    unstash "04_vit.log"
                    unstash "05_sdiff.log"
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

//launch amd-develop branch daily at 17:00 UT in FULL_QA mode 
CRON_SETTINGS = BRANCH_NAME == "amd-develop" ? '''0 17 * * * % RUN_FULL_QA=true''' : ""

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
        booleanParam(
            name: "RUN_FULL_QA",
            defaultValue: false,
            description: "Select whether to run small set of performance tests (default) or full QA")
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
            when {
                beforeAgent true
                expression { params.RUN_FULL_QA.toBoolean() }
            }
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

