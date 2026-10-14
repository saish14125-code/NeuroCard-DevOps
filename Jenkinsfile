pipeline {
    agent any

    stages {
        stage('Install Dependencies') {
            steps {
                dir('credit-ai-ui') {
                    bat 'npm install'
                }
            }
        }

        stage('Build React Application') {
            steps {
                dir('credit-ai-ui') {
                    bat 'set CI=false&& npm run build'
                }
            }
        }

        stage('Deploy to Nginx') {
            steps {
                bat '''
                    if not exist "C:\\nginx\\html" mkdir "C:\\nginx\\html"
                    robocopy "credit-ai-ui\\build" "C:\\nginx\\html" /E
                    if %ERRORLEVEL% GEQ 8 exit /b %ERRORLEVEL%
                '''
            }
        }
    }

    post {
        success {
            echo 'NeuroCard frontend deployed successfully to Nginx!'
        }
        failure {
            echo 'Pipeline failed. Check the console output for details.'
        }
    }
}