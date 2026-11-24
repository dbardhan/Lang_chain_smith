pipeline {
    agent any
    options {
        // Prevent two builds of the same job from running concurrently.
        disableConcurrentBuilds()
        // Add timestamps to Jenkins console output.
        timestamps()
    }
    stages {
        stage('Environment Info') {
            steps {
                echo 'Checking CI environment...'
                sh '''
                    echo "Branch: ${BRANCH_NAME:-unknown}"
                    echo "Workspace: ${WORKSPACE}"
                    git --version
                    python3.13 --version
                '''
            }
        }
        stage('Setup Python Environment') {
            steps {
                echo 'Creating Python virtual environment...'
                sh '''
                    rm -rf .venv
                    python3.13 -m venv .venv
                    . .venv/bin/activate
                    python --version
                    python -m pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }
        stage('Pre-commit') {
            steps {
                echo 'Running pre-commit checks...'
                sh '''
                    . .venv/bin/activate
                    pre-commit --version
                    pre-commit run --all-files
                '''
            }
        }
        stage('Pytest') {
            steps {
                echo 'Running unit tests...'
                sh '''
                    . .venv/bin/activate
                    pytest --version
                    pytest -v
                '''
            }
        }
    }
    post {
        success {
            echo '===================================='
            echo 'CI PASSED'
            echo 'Pre-commit: PASSED'
            echo 'Pytest: PASSED'
            echo '===================================='
        }
        failure {
            echo '===================================='
            echo 'CI FAILED'
            echo 'Check Jenkins console output.'
            echo '===================================='
        }
        always {
            echo 'Jenkins CI pipeline completed.'
        }
    }
}
