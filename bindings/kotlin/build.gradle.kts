plugins {
    kotlin("jvm") version "2.0.21"
}

group = "ai.meshllm"
version = "0.1.0"

repositories {
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.7.3")
    testImplementation("junit:junit:4.13.2")
    testImplementation("io.mockk:mockk:1.13.8")
}

// Task to build native libraries for all Android ABIs
val buildNativeLibs by tasks.registering(Exec::class) {
    description = "Build mesh-api-ffi shared libraries for all Android ABIs"
    group = "build"

    val repoRoot = rootProject.projectDir.parentFile.parentFile
    val ndkHome = System.getenv("ANDROID_NDK_HOME")
        ?: "${System.getProperty("user.home")}/Library/Android/sdk/ndk/26.3.11579264"

    val rustupRustc = System.getenv("RUSTC")
        ?: "${System.getProperty("user.home")}/.rustup/toolchains/stable-aarch64-apple-darwin/bin/rustc"

    environment("ANDROID_NDK_HOME", ndkHome)
    environment("ANDROID_NDK_ROOT", ndkHome)
    environment("RUSTC", rustupRustc)

    commandLine(
        "bash", "-c",
        """
        set -e
        cd ${repoRoot}

        # Build for arm64-v8a (RUSTC override needed: Homebrew rustc shadows rustup)
        ANDROID_NDK_HOME=${ndkHome} \
        cargo ndk -t arm64-v8a build --release -p mesh-api-ffi --no-default-features

        # Build for armeabi-v7a
        ANDROID_NDK_HOME=${ndkHome} \
        cargo ndk -t armeabi-v7a build --release -p mesh-api-ffi --no-default-features

        # Build for x86_64
        ANDROID_NDK_HOME=${ndkHome} \
        cargo ndk -t x86_64 build --release -p mesh-api-ffi --no-default-features

        # Copy to jniLibs
        cp target/aarch64-linux-android/release/libmesh_ffi.so bindings/kotlin/src/main/jniLibs/arm64-v8a/
        cp target/armv7-linux-androideabi/release/libmesh_ffi.so bindings/kotlin/src/main/jniLibs/armeabi-v7a/
        cp target/x86_64-linux-android/release/libmesh_ffi.so bindings/kotlin/src/main/jniLibs/x86_64/
        """.trimIndent()
    )

    outputs.files(
        "${projectDir}/src/main/jniLibs/arm64-v8a/libmesh_ffi.so",
        "${projectDir}/src/main/jniLibs/armeabi-v7a/libmesh_ffi.so",
        "${projectDir}/src/main/jniLibs/x86_64/libmesh_ffi.so"
    )
}

tasks.named("build") {
    dependsOn(buildNativeLibs)
}

// Assemble a distributable AAR artifact (ZIP format) containing:
//   classes.jar              — compiled Kotlin classes
//   jni/<abi>/libmesh_ffi.so — native shared libraries
//   consumer-proguard-rules.pro
//   AndroidManifest.xml      — minimal manifest required by AAR spec
val assembleAar by tasks.registering(Zip::class) {
    description = "Assemble AAR artifact with native libs and consumer ProGuard rules"
    group = "build"

    dependsOn("jar")

    archiveFileName.set("meshllm.aar")
    destinationDirectory.set(layout.buildDirectory.dir("outputs/aar"))

    // Compiled Kotlin classes, renamed to the standard AAR entry name
    from(tasks.named<Jar>("jar")) {
        rename { "classes.jar" }
    }

    // Native shared libraries under jni/<abi>/
    from("src/main/jniLibs") {
        into("jni")
    }

    // Consumer ProGuard rules consumed by downstream Android projects
    from("consumer-proguard-rules.pro")

    // Minimal AndroidManifest required by the AAR format
    from("src/main/AndroidManifest.xml")
}
